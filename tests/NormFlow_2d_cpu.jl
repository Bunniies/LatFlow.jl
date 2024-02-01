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
    lattice_shape = (8,8)
    m2 = -4.0
    lambda = 8.0
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
    epochs::Int64 = 10
    batch_size::Int64 = 64
    eta_lr::Float64 = 0.001
end

@kwdef struct HyperParams
    dp::DeviceParams = DeviceParams(0)
    ap::ActionParams = ActionParams()
    mp::ModelParams = ModelParams()
    tp::TrainingParams = TrainingParams()
end

struct Phi4ScalarAction
    m2::Float32
    lambda::Float32
end

function (action::Phi4ScalarAction)(cnfg)
    U = cnfg .^2 .* action.m2 .+ cnfg .^4 .* action.lambda
    dim = ndims(cnfg) 
    kin = sum(cnfg .* ( 2 * cnfg - circshift(cnfg, -Flux.onehot(k,1:dim)) - circshift(cnfg, Flux.onehot(k,1:dim)) ) for k in 1:dim-1)
    action_sum = sum(kin .+ U, dims=1:dim-1)

    return dropdims(action_sum, dims=Tuple((1:dim-1)))
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
    x_pr_device = x_pr |> device
    xout, logq = affine_layer((x_pr_device, logq_prec ))
    return xout, logq
end

function build_cnn(mp::ModelParams)  
    seed = mp.seed
    ch_tot = [mp.inCh, mp.hidden_ch..., mp.outCh]
    kernel_size = tuple(fill(mp.kernel_size,2)...)

    net = []
    for kch in eachindex(ch_tot[1:end-2])
        if !mp.use_bn
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], leakyrelu, pad=0, stride=1))
        else
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], pad=0, stride=1))
            push!(net, BatchNorm(ch_tot[kch+1], leakyrelu))
        end
    end
    
    if mp.use_tanh_last
        ch = ch_tot[end-1]
        ch_last = ch_tot[end]
        if !mp.use_bn
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch=>ch_last, tanh, pad=0, stride=1))
        else
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Chain(Conv(kernel_size, ch=>ch_last, pad=0, stride=1), BatchNorm(ch_last, tanh)))
        end
    end
    return net
end

function create_affine_layers(hp::HyperParams)
    n_layers = hp.mp.n_layers
    device = hp.dp.device
    
    couplings = []
    for k in 0:(n_layers-1)
        net = build_cnn(hp.mp)
        mask = freezing_mask(hp.ap.lattice_shape, mod(k,2))
        tmp_coupling = AffineCoupling(Chain(net...), mask) 
        push!(couplings, tmp_coupling) 
    end
    affine_layers = Chain(couplings...) |> f32 |> device
    return affine_layers
end

dp = DeviceParams(0)
ap = ActionParams(lattice_shape=(8,8))
mp = ModelParams()
tp = TrainingParams()
hp = HyperParams(dp, ap, mp, tp)

##

## Test affine layers and plotting untrained configs

layer = create_affine_layers(hp)
prior = Normal{Float32}(0.f0, 1.f0)

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


using TimerOutputs

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
        # ts = @timed begin
        @timeit "Training" begin
            
            for kk in 1:iterations

                x_pr = rand(prior, hp.ap.lattice_shape..., batch_size ) 
                logq_prec = sum(logpdf.(prior, x_pr), dims=1:ndims(x_pr)-1) |> device
                x_pr_dev = x_pr |> device

                grads = Flux.gradient(ps) do 
                    x_out, logq_ = affine_layers((x_pr_dev, logq_prec)) 
                    logq = vcat(logq_)
                    logp = - action(x_out)
                    loss = compute_KL_div(logp, logq |> device)
                end
                Flux.Optimise.update!(opt, ps, grads)
            end
        end 

        # test mode 
        Flux.testmode!(affine_layers)
        x_out, logq = evolve_prior_with_flow(prior, affine_layers, batchsize=batch_size, lattice_shape=hp.ap.lattice_shape, device=device)
        #logq = dropdims(logq, dims=(1,ndims(logq)-1))
        logq = vcat(logq...)

        logp = -action(x_out)
        loss = compute_KL_div(logp, logq |> device)
        @show loss
        ess = compute_ESS(logp, logq |> device)
        @show ess 
        push!(history[!,"epochs"], epoch)
        #push!(history[!,"timing"], ts.time)
        push!(history[!,"timing"], 0.0)
        push!(history[!,"loss"], loss)
        push!(history[!,"ess"], ess)
        push!(history[!,"acceptance_rate"], 0.0)
    end
    return affine_layers, history
end

##
layer = create_affine_layers(hp)
prior = Normal{Float32}(0.f0, 1.f0)

action = Phi4ScalarAction(hp.ap.m2, hp.ap.lambda)
x_out, logq = evolve_prior_with_flow(prior, layer, batchsize=hp.tp.batch_size, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
logp = -action(x_out)
compute_KL_div(logp, vcat(logq...))

## Training
prior = Normal{Float32}(0.f0, 1.f0)
action = Phi4ScalarAction(hp.ap.m2, hp.ap.lambda)
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
action = Phi4ScalarAction(hp.ap.m2, hp.ap.lambda)

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
        x_out_, logq = evolve_prior_with_flow(prior, layer, batchsize=batchsize, lattice_shape=lattice_shape, device=device)
        logq = vec(logq) |> cpu
        logp = -action(x_out_) |> cpu
        x_out = x_out_ |> cpu

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

nsamples= 8192
mcmc_hist = build_mcmc(prior, affine_layer, action, batchsize=hp.tp.batch_size, nsamples=nsamples, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device )
mcmc_hist[!, "accepted"] |> mean

## check susceptibility
vec_config = mcmc_hist[!, "config"]
cnfg = Flux.MLUtils.batch(mcmc_hist[!, "config"][512:end])

C = 0.0
for x in 1:hp.ap.lattice_shape[1]
    for y in 1:hp.ap.lattice_shape[2]
        C = C .+ cnfg .* circshift(cnfg, (-x, -y))
    end
end
X = mean(C, dims=(1,2))

mean(X )
std(X)

## copying Gomalazing flow for susceptibility


function green(cfgs::AbstractArray{T, N}, offsetX) where {T, N}
    # shifts = (broadcast(-, offsetX)..., 0)
    shifts = broadcast(-, offsetX)
    batch_dim = N
    lattice_shape = (size(cfgs, i) for i in 1:N-1)
    cfgs_offset = circshift(cfgs, shifts)
    m_corr = mean(cfgs .* cfgs_offset, dims=batch_dim)
    # m_corr = uwreal(m_corr_uw, "ens")
    m = mean(cfgs, dims=batch_dim)
    m_offset = mean(cfgs_offset, dims=batch_dim)
    V = prod(lattice_shape)
    Gc = sum(m_corr .- m .* m_offset)/V
    return Gc
end

function my_green(cnfg::AbstractArray{T,N}, offsetX) where {T,N}
    shifts = broadcast(-, offsetX)
    batch_dim = N
    lattice_shape = (size(cnfg, i) for i in 1:N-1)
    cnfg_offset = circshift(cnfg, shifts)
    m_corr = Matrix{uwreal}(undef, Tuple(lattice_shape))
    m = Matrix{uwreal}(undef, Tuple(lattice_shape))
    m_offset = Matrix{uwreal}(undef, Tuple(lattice_shape))
    for t in 1:Tuple(lattice_shape)[1]
        for x in 1:Tuple(lattice_shape)[2]
            m_corr[t,x]   = uwreal(cnfg[t,x,:] .* cnfg_offset[t,x,:] .|> Float64, "ens")
            m[t,x]        = uwreal(cnfg[t,x,:] .|> Float64 , "ens")
            m_offset[t,x] = uwreal(cnfg_offset[t,x,:] .|> Float64, "ens")
        end
    end
    V = prod(lattice_shape)
    Gc = sum(m_corr .- m.*m_offset)/V
end


function mfGC(cnfg, t)
    lattice_shape = size(cnfg)[begin:end-1]
    acc = 0.0
    for s in collect(Iterators.product((1:l for l in lattice_shape)...))
        acc += green(cnfg, (s..., t))
    end
    return acc
end

function susceptibility(cfgs)
    lattice_shape = (cfgs |> size)[begin:end-1]
    acc = 0.0f0
    for s in collect(Iterators.product((1:l for l in lattice_shape)...))
        acc += my_green(cfgs, s)
    end
    return acc
end

two_pt = []
for t in 1:hp.ap.lattice_shape[1]
    y = mfGC(cnfg, t)
    push!(two_pt, y)
end

## my own green
using ADerrors

function green_2pt(cnfg::AbstractArray{T,N}) where {T,N}
    batch_dim = N
    L = size(cnfg,1)
    Gx = Vector{uwreal}(undef, L)

    for x 





end

##
xxx = rand(1000)

hist(xxx, bins=100)
display(gcf())
close("all")