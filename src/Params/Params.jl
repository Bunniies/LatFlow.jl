module Params

using Base:@kwdef
using Flux, CUDA, UnPack, TimerOutputs


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
export DeviceParams


@kwdef struct ActionParams
    lattice_shape = (8,8)
    m2 = -4.0
    lambda = 8.0
    beta = 2.0
end
export ActionParams


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
export ModelParams


@kwdef struct TrainingParams
    iterations::Int64 = 100
    epochs::Int64 = 100
    batch_size::Int64 = 64
    eta_lr::Float64 = 0.002
end
export TrainingParams

@kwdef struct HyperParams
    dp::DeviceParams = DeviceParams(0)
    ap::ActionParams = ActionParams()
    mp::ModelParams = ModelParams()
    tp::TrainingParams = TrainingParams()
end
export HyperParams

include("workspace.jl")
export NFworkspace




end