using Flux, CUDA, Distributions
using Base:@kwdef
using PyPlot
using UnPack
using ProgressMeter
using DataFrames
using Random



struct DeviceParams
    id::Int
    device::Function 
    function DeviceParams(id)
        if id >= 0 && CUDA.functional()
            CUDA.device!(id)
            device = Flux.gpu
            @info "Chosen device: GPU with id=$(id)"
        else
            @warn "You've set GPU with id = $id, but CUDA.functional() is $(CUDA.functional())"
            @info "Chosen device: CPU"
            id = -1
            device = Flux.cpu
        end
        new(id, device)
    end
end

@kwdef struct ActionParams
    lattice_shape = (4,4)
    m = 1.0
    lambda = 1.0
    beta = 2.0
end
#ActionParams(L, m, lambda, beta) = ActionParams((L,L), m, lambda, beta)
#ap = ActionParams(4, 1.0, 2.0, 2.0)
ActionParams(lattice_shape=(8,8))

@kwdef struct ModelParams
    seed::Int64 = 1994
    inCh::Int64 = 1
    outCh::Int64 = 2
    n_layers::Int64 = 16
    hidden_ch::Vector{Int64} = [8,8]
    kernel_size::Int64 = 3
    use_tanh_last::Bool = true
    use_bn::Bool = false
end

@kwdef struct TrainingParams
    iterations::Int64 = 100
    epochs::Int64 = 100
    batch_size::Int64 = 64
    eta_lr::Float64 = 0.002
end

@kwdef struct HyperParams
    dp::DeviceParams = DeviceParams(0)
    ap::ActionParams = ActionParams()
    mp::ModelParams = ModelParams()
    tp::TrainingParams = TrainingParams()
end

struct Phi4ScalarAction
    m::Float32
    lambda::Float32
end

function (action::Phi4ScalarAction)(cnfg)
    U = cnfg .^2 .* action.m^2 .+ cnfg .^4 .* action.lambda
    dim = ndims(cnfg) 
    # kin = zeros(size(cnfg))
    kin = sum(cnfg .* ( 2 * cnfg - circshift(cnfg, -Flux.onehot(k,1:dim)) - circshift(cnfg, Flux.onehot(k,1:dim)) ) for k in 1:dim-1)
    # for k in 1:(dim-1)
        # kin .+= cnfg .* ( 2 * cnfg - circshift(cnfg, -Flux.onehot(k,1:dim)) - circshift(cnfg, Flux.onehot(k,1:dim)) )
    # end
    action_sum = sum(kin .+ U, dims=1:dim-1)
    # action_sum = kin .+ U

    return dropdims(action_sum, dims=Tuple((1:dim-1)))
end

function (action::Phi4ScalarAction)(cfgs)
    potential = @. action.m^2 * cfgs ^ 2 + action.lambda * cfgs ^ 4
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    k1 = sum(2cfgs .^ 2 for μ in 1:Nd)
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = potential .+ k1 .- k2 .- k3
    dropdims(
        sum(action_density, dims=1:Nd),
        dims=Tuple(1:Nd)
    )
end

function freezing_mask(shape, parity)
    mask = ones(shape) .- parity
    mask[1:2:end, 1:2:end] .= parity
    mask[2:2:end, 2:2:end] .= parity
    return mask
end


struct AffineCoupling{Model, Mask}
    nn::Model
    mask::Mask
end

Flux.@functor AffineCoupling

function (layer::AffineCoupling)(cnfg)
    x_pr = cnfg[1]
    logq_prec = cnfg[2]
    # println("mask size: ", size(layer.mask))
    # println("cnfg size: ", size(cnfg))
    x_pr_frozen =  layer.mask .* x_pr
    x_pr_active = x_pr .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(x_pr_frozen, dims=ndims(x_pr_frozen)))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    fx = @. (1 - layer.mask) * t +  x_pr_active * exp(s) + x_pr_frozen
    logJ = sum((1 .- layer.mask) .* s, dims=1:(ndims(s)-1)) 
    return fx, logq_prec .- logJ
end
forward(layer::AffineCoupling, cnfg) = layer(cnfg)

function reverse(layer::AffineCoupling, fcnfg)
    fcnfg_frozen = fcnfg .* layer.mask
    fcnfg_active = fcnfg .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(fcnfg_frozen, dims=ndims(fcnfg_frozen)))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    x = @. (fcnfg_active - (1 - layer.mask) * t) * exp(-s) + fcnfg_frozen
    logJ = sum((1 .- layer.mask) .* (.-s), dims=1:(ndims(s)-1)) 
    return x, logJ
end

function evolve_prior_with_flow(prior, affine_layer; batchsize, lattice_shape, device)
    x_pr =  rand(prior, lattice_shape..., batchsize ) 
    logq_prec = sum(logpdf.(prior, x_pr), dims=(1:ndims(x_pr)-1)) |> device
    xout, logq = affine_layer((x_pr |> device, logq_prec ))
    return xout, logq
end

function build_cnn(mp::ModelParams)  
    seed = mp.seed
    ch_tot = [mp.inCh, mp.hidden_ch..., mp.outCh]
    kernel_size = tuple(fill(mp.kernel_size,2)...)

    net = []
    for kch in eachindex(ch_tot[1:end-1])
        if !mp.use_bn
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], leakyrelu, pad=1, stride=1))
        else
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], pad=1, stride=1))
            push!(net, BatchNorm(ch_tot[kch+1], leakyrelu))
        end
    end
    
    if mp.use_tanh_last
        ch = ch_tot[end-1]
        ch_last = ch_tot[end]
        if !mp.use_bn
            net[end] = Conv(kernel_size, ch=>ch_last, tanh, pad=1, stride=1)
        else
            net[end] = Chain(Conv(kernel_size, ch=>ch_last, pad=1, stride=1), BatchNorm(ch_last, tanh))
        end
    end
    return net
end



# function create_affine_layers(hp::HyperParams)
    # n_layers = hp.mp.n_layers
    # device = hp.dp.device
    # 
    # couplings = []
    # for k in 0:(n_layers-1)
        # net = build_cnn(hp.mp)
        # mask = freezing_mask(hp.ap.lattice_shape, mod(k,2))
        # tmp_coupling = AffineCoupling(Chain(net...), mask) 
        # push!(couplings, tmp_coupling) 
    # end
    # affine_layers = Chain(couplings...) |> f32 |> device
    # return affine_layers
# end

function create_affine_layers(hp::HyperParams)

    n_layers = hp.mp.n_layers
    device = hp.dp.device
    module_list = []
    for i in 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [hp.mp.inCh, hp.mp.hidden_ch..., hp.mp.outCh]
        padding = hp.mp.kernel_size ÷ 2
        net = []
        for kch in eachindex(channels[1:end-1])
            c = channels[kch]
            c_next = channels[kch+1]
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            k = 1/(c * 3 * 3)
            W = rand(Uniform(-√k, √k), 3, 3, c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            #push!(net, mycircular)
            push!(net, Conv(W, b, leakyrelu, pad=0))
        end
        if hp.mp.use_tanh_last
            c = channels[end-1]
            c_next = channels[end]
            k = 1/(c * 3 * 3)
            W = rand(Uniform(-√k, √k), 3, 3, c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            net[end] = Conv(W, b, tanh, pad=0)
        end
        mask = freezing_mask(hp.ap.lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list, coupling)
    end
    layer = Chain(module_list...) |> f32 |> device
    return layer

end

dp = DeviceParams(0)
ap = ActionParams(lattice_shape=(8,8))
mp = ModelParams()
tp = TrainingParams()
hp = HyperParams(dp, ap, mp, tp)

##

prior = Normal{Float32}(0.f0, 1.f0)
x_pr =  rand(prior, hp.ap.lattice_shape..., hp.tp.batch_size ) 
mask = freezing_mask(hp.ap.lattice_shape, 0)
mask .* x_pr

## Test affine layers and plotting untrained configs

layer = create_affine_layers(hp)
x, logq = evolve_prior_with_flow(prior, layer, batchsize=hp.tp.batch_size, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
x = x |> cpu

ig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end

display(gcf())
close("all")

##

function compute_KL_div(logp, logq)
    return mean(logq .- logp)
end

function compute_ESS(logp, logq)
    log_diff = logp - logq
    log_ess = 2*logsumexp(log_diff) - logsumexp(2*log_diff)
    return exp(log_ess) / length(log_diff)
end

function get_training_param(layer)
    ps = Flux.params(layer)
    for k in eachindex(layer)
        delete!(ps, layer[k].mask)
    end
    return ps
end



function train(hp::HyperParams, action)
    dp = hp.dp
    ap = hp.ap
    mp = hp.mp
    tp = hp.tp
    @info "Model setup"
    @unpack iterations, epochs, batch_size, eta_lr = hp.tp
    device = hp.dp.device

    affine_layers = create_affine_layers(hp)
    ps = get_training_param(affine_layers)

    opt = Adam(hp.tp.eta_lr)
    #action = Phi4ScalarAction(hp.ap.m, hp.ap.lambda)
    prior = Normal{Float32}(0.f0, 1.f0)

    history = DataFrame(
        "epochs"          => Int[],
        "loss"            => Float64[],
        "ess"             => Float64[],
        "timing"          => Float64[],
        "acceptance_rate" => Float64[]
    )


    @info "Training model"

    @showprogress for epoch in 1:epochs
        @info "\n epoch=$epoch"
        
        # train mode 
        Flux.trainmode!(affine_layers)
        ts = @timed begin
            for kk in 1:iterations

                @info "     iterations=$(kk)"
                x_pr_dev = rand(prior, hp.ap.lattice_shape..., hp.tp.batch_size ) 
                logq_prec = sum(logpdf.(prior, x_pr_dev), dims=1:ndims(x_pr_dev)-1) |> device
                x_pr = x_pr_dev |> device

                grads = Flux.gradient(ps) do 
                    x_out, logq = affine_layers((x_pr, logq_prec)) 
                    logq = vcat(logq...)
                    logp = - action(x_out)
                    loss = compute_KL_div(logp, logq)
                end
                Flux.Optimise.update!(opt, ps, grads)
            end
        end 

        # test mode 
        Flux.testmode!(affine_layers)
        # prior = sampleNorm(hp.tp.batch_size) |> device
        x_out, logq = evolve_prior_with_flow(prior, affine_layers, batchsize=hp.tp.batch_size, lattice_shape=hp.ap.lattice_shape, device=device)
        #logq = dropdims(logq, dims=(1,ndims(logq)-1))
        logq = vcat(logq...)

        logp = -action(x_out)
        loss = compute_KL_div(logp, logq)
        @show loss
        ess = compute_ESS(logp, logq)
        @show ess 
        push!(history[!,"epochs"], epoch)
        push!(history[!,"timing"], ts.time)
        push!(history[!,"loss"], loss)
        push!(history[!,"ess"], ess)
        push!(history[!,"acceptance_rate"], 0.0)
    end
    return affine_layers, history
end

##
action = Phi4ScalarAction(hp.ap.m, hp.ap.lambda)

affine_layer, train_hist = train(hp, action)

## check trained model

x, logq = evolve_prior_with_flow(prior, affine_layer, batchsize=hp.tp.batch_size, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
x = x |> cpu

ig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end

display(gcf())
close("all")

##
x, logq = evolve_prior_with_flow(prior, affine_layer, batchsize=1024, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
x = cpu(x)
S_eff = -logq |> cpu
action = Phi4ScalarAction(hp.ap.m, hp.ap.lambda)

S = action(x)
fit_b = mean(S) - mean(S_eff)
@show fit_b
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20,
    #range=[[5, 35], [-5, 25]]
)

xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
display(gcf())
close("all")


## implementing markov chain

function build_mcmc(prior, layer, action; batchsize, nsamples, lattice_shape, device=cpu, seed=3430)

    rng = MersenneTwister(seed)
    mcmc_hist = DataFrame(
        "logp"     => Float32[],
        "logq"     => Float32[],
        "config"   => Array{Float32, 2}[],
        "accepted" => Bool[]
    )

    counter = 0
    @showprogress for _ in 1:round(Int, nsamples/batchsize)
        x_out, logq = evolve_prior_with_flow(prior, layer, batchsize=batchsize, lattice_shape=lattice_shape, device=device)
        logq = vec(logq) |> cpu
        logp = -action(x_out) |> cpu
        x_out = x_out |> cpu

        for k in 1:batchsize
            x_proposed = x_out[:,:,k]
            logq_proposed = logq[k]
            logp_proposed = logp[k]

            if isempty(mcmc_hist[!,"logp"])
                acc = true
            else
                logq_prev = mcmc_hist[!, "logq"][end]
                logp_prev = mcmc_hist[!, "logp"][end]
                p_acc = exp((logp_proposed - logq_proposed) - (logp_prev - logq_prev))
                p_acc = min(one(eltype(p_acc)), p_acc)
                coin = rand(rng, eltype(p_acc))
                println("pacc: ", p_acc, " coin: ", coin)
                if coin < p_acc
                    acc = true
                else
                    acc = false
                    x_proposed = mcmc_hist[!, "config"][end]
                    logq_proposed = logq_prev
                    logp_proposed = logp_prev
                end
            end
            push!(mcmc_hist[!, "logp"], logp_proposed)
            push!(mcmc_hist[!, "logq"], logq_proposed)
            push!(mcmc_hist[!, "config"], x_proposed)
            push!(mcmc_hist[!, "accepted"], acc)
        end
        counter += batchsize
        if counter >= nsamples
            break
        end
    end
    return mcmc_hist
end

nsamples= 8092
mcmc_hist = build_mcmc(prior, affine_layer, action, batchsize=hp.tp.batch_size, nsamples=nsamples, lattice_shape=hp.ap.lattice_shape )
mcmc_hist[!, "accepted"] |> mean



