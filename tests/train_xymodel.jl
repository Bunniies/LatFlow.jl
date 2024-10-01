using Flux, CUDA, Distributions
using Base:@kwdef
using PyPlot
using UnPack
using ProgressMeter
using DataFrames
using Random
using TimerOutputs

# Hyper Parameters

struct DeviceParams
    id::Int
    device::Function 
    function DeviceParams(id)
        if id >= 0 && CUDA.functional()
            CUDA.device!(id)
            device = Flux.gpu
            @info "Chosen device: GPU with id=$(id)"
        else
            if id >= 0
                @warn "You've set GPU with id = $id, but CUDA.functional() is $(CUDA.functional())"
            end
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

# XY action 

struct XYmodelAction
    beta::Float32
end

function (action::XYmodelAction)(cnfg)
    dim = ndims(cnfg)
    action_density = sum( cos.(cnfg - circshift(cnfg, Flux.onehot(k,1:dim))) for k in 1:dim-1 )
    act = sum(action_density, dims=1:dim-1)
    return  -1 .* action.beta  .* dropdims(act, dims=Tuple((1:dim-1))) 
end

function freezing_mask(shape, parity)
    mask = ones(Float32, shape) .- Float32(parity)
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
    x_pr_frozen =  layer.mask .* x_pr
    x_pr_active = x_pr .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(x_pr_frozen, dims=ndims(x_pr_frozen)))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    println("size(x_out): ", size(x_out))
    println("size(s): ", size(s))
    println("size(t): ", size(t))
    
    fx = @. (1 - layer.mask) * t +  x_pr_active * exp(s) + x_pr_frozen
    logJ = sum((1 .- layer.mask) .* s, dims=1:(ndims(s)-1)) 
    return fx, logq_prec .- logJ
end
forward(layer::AffineCoupling, cnfg) = layer(cnfg)