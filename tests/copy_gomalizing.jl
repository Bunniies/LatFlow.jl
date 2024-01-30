using Random

#using PyCall
using PyPlot
using Distributions
using Flux
#using EllipsisNotation
#using IterTools
using LaTeXStrings
using ProgressMeter
using Base:@kwdef

using CUDA
use_cuda = true

if use_cuda && CUDA.functional()
    device_id = 0 # 0, 1, 2 ...
    CUDA.device!(device_id)
    device = gpu
    @info "Training on GPU"
else
    device = cpu
    @info "Training on CPU"
end

struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end

function calc_action(action::ScalarPhi4Action, cfgs)
    potential = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
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

function (action::ScalarPhi4Action)(cfgs)
    calc_action(action, cfgs)
end

function make_checker_mask(shape, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end] .= parity
    checker[(begin+1):2:end, (begin+1):2:end] .= parity
    return checker
end

function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end

function apply_affine_flow_to_prior(prior, affine_coupling_layers; batchsize)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf.(prior, x), dims=(1:ndims(x)-1)) |> device
    xout, logq = affine_coupling_layers((x |> device, logq_))
    return xout, logq
end

function mycircular(Y)
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under

    # calc Z_top
    Y_e_c = Y[end:end,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)

    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:]
    Y_main_r = Y[:,end:end,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    cat(Z_top, Z_main, Z_bottom, dims=1)
end

struct AffineCoupling{A, B}
    net::A
    mask::B
end

Flux.@functor AffineCoupling


function (model::AffineCoupling)(x_pair_loghidden)
    x = x_pair_loghidden[begin]
    loghidden = x_pair_loghidden[end]
    x_frozen = model.mask .* x
    x_active = (1 .- model.mask) .* x
    # (inW, inH, inB) -> (inW, inH, 1, inB) # by Flux.unsqueeze(*, 3)
    net_out = model.net(Flux.unsqueeze(x_frozen, 3))
    s = @view net_out[:, :, 1, :] # extract feature from 1st channel
    t = @view net_out[:, :, 2, :] # extract feature from 2nd channel
    fx = @. (1 - model.mask) * t + x_active * exp(s) + x_frozen
    logJ = sum((1 .- model.mask) .* s, dims=1:(ndims(s)-1))
    return (fx, loghidden .- logJ)
end

# alias
forward(model::AffineCoupling, x_pair_loghidden) = model(x_pair_loghidden)

function reverse(model::AffineCoupling, fx)
    fx_frozen = model.mask .* fx
    fx_active = (1 .- model.mask) .* fx
    net_out = model(fx_frozen)
    return net_out
end


const L = 8
const lattice_shape = (L, L)
const M2 = -4.
const lam = 8 #lam = 8.
const phi4_action = ScalarPhi4Action(M2, lam)

const n_layers = 16
const hidden_sizes = [8, 8]
const kernel_size = 3
const inC = 1
const outC = 2
const use_final_tanh = true

const prior = Normal{Float32}(0.f0, 1.f0)

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

function create_layer()
    module_list = []
    for i ∈ 0:(n_layers-1)
        parity = mod(i, 2)
        net = build_cnn(ModelParams())
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list, coupling)
    end
    layer = Chain(module_list...) |> f32 |> device
    ps = Flux.params(layer)
    for i in 0:(n_layers-1)
        delete!(ps, layer[i+1].mask)
    end
    return layer, ps
end

function create_layer()
    module_list = []
    for i ∈ 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        padding = kernel_size ÷ 2
        net = []
        for (c, c_next) ∈ pairwise(channels)
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            k = 1/(c * 3 * 3)
            W = rand(Uniform(-√k, √k), 3, 3, c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            push!(net, mycircular)
            push!(net, Conv(W, b, leakyrelu, pad=0))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            k = 1/(c * 3 * 3)
            W = rand(Uniform(-√k, √k), 3, 3, c, c_next)
            b = rand(Uniform(-√k, √k), c_next)
            net[end] = Conv(W, b, tanh, pad=0)
        end
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list, coupling)
    end
    layer = Chain(module_list...) |> f32 |> device
    ps = Flux.params(layer)
    for i in 0:(n_layers-1)
        delete!(ps, layer[i+1].mask)
    end
    return layer, ps
end

layer, ps = create_layer()

x, logq = apply_affine_flow_to_prior(prior, layer; batchsize=64)
x = x |> cpu
fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
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
calc_dkl(logp, logq) = mean(logq .- logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2*logsumexp(logw) - logsumexp(2*logw)
    ess_per_cfg = exp(log_ess) / length(logw)
    return ess_per_cfg
end

reversedims(inp::AbstractArray{<:Any, N}) where {N} = permutedims(inp, N:-1:1)


function train()
    layer, ps = create_layer()
    batchsize = 64
    n_era = 10
    epochs = 100

    base_lr = 0.001f0
    opt = Adam(base_lr)

    for era in 1:n_era
        @showprogress for e in 1:epochs
            x = rand(prior, lattice_shape..., batchsize)
            logq_in = sum(logpdf.(prior, x), dims=(1:ndims(x)-1)) |> device
            xin = x |> device
            gs = Flux.gradient(ps) do
                xout, logq_out = layer((xin, logq_in))
                logq = dropdims(
                    logq_out,
                    dims=Tuple(1:(ndims(logq_out)-1))
                )
                logp = -calc_action(phi4_action, xout)
                loss = calc_dkl(logp, logq)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        x, logq_ = apply_affine_flow_to_prior(prior, layer; batchsize)
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_)-1))
        )

        logp = -calc_action(phi4_action, x)
        loss = calc_dkl(logp, logq)
        @show loss
        @show "loss per site" loss/prod(lattice_shape)
        ess = compute_ess(logp, logq)
        @show ess
    end
    layer
end

layer = train();

##
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize=64)
x = x |> cpu
fig, ax = plt.subplots(4, 4, dpi=125, figsize=(4, 4))
for i in 1:4
    for j in 1:4
        ind = 4i + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end
display(gcf())
close("all")

##
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize=1024)
x = cpu(x)
S_eff = -logq |> cpu
S = calc_action(phi4_action, x)
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

##

function make_mcmc_ensamble(layer, prior, action; batchsize, nsamples)
    history=(x=Matrix{Float32}[], logq=Float32[], logp=Float32[], accepted=Bool[])
    c = 0
    @showprogress for _ in 1:(nsamples÷batchsize + 1)
        x_device, logq_ = apply_affine_flow_to_prior(prior, layer; batchsize)
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_)-1))
        ) |> cpu
        logp = -phi4_action(x_device) |> cpu
        x = x_device |> cpu
        for b in 1:batchsize
            new_x = x[:,:, b]
            new_logq = logq[b]
            new_logp = logp[b]
            if isempty(history[:logp])
                accepted = true
            else
                last_logp = history[:logp][end]
                last_logq = history[:logq][end]
                last_x = history[:x][end]
                p_accept = exp((new_logp - new_logq) - (last_logp - last_logq))
                p_accept = min(one(p_accept), p_accept)
                draw = rand()
                if draw < p_accept
                    accepted = true
                else
                    accepted = false
                    new_x = last_x
                    new_logp = last_logp
                    new_logq = last_logq
                end
            end
            # update history
            push!(history[:logp], new_logp)
            push!(history[:logq], new_logq)
            push!(history[:x], new_x)
            push!(history[:accepted], accepted)
        end
        c += batchsize
        if c >= nsamples
            break
        end
    end
    history
end

ensamble_size = 8192
history = make_mcmc_ensamble(layer, prior, phi4_action; batchsize=64, nsamples=ensamble_size);
history[:accepted] |> mean

##
cnfg = Flux.MLUtils.batch(history[:x][512:end])

C = 0.0
for x in 1:L
    for y in 1:L
        C = C .+ cnfg .* circshift(cnfg, (-x, -y))
    end
end
X = mean(C, dims=(1,2))

mean(X )
std(X)