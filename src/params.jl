struct DeviceParams
    id::Int
    device::Function 
    function DeviceParams(id)
        if id >= 0 && CUDA.functional()
            CUDA.device!(id)
            device = Flux.gpu
            @info "Device: GPU with id=$(id)"
        else
            if id > 0
                @warn "You've set GPU with id = $id, but CUDA.functional() is $(CUDA.functional())"
            end
            @info "Device: CPU"
            id = -1
            device = Flux.cpu
        end
        new(id, device)
    end
end

@kwdef struct ActionParams
    L::Int64=8
    dim::Vector{Int64}=[L,L]
    m = 1.0
    lambda = 1.0
end
ActionParams(L, m, lambda) = ActionParams(L, [L, L], m, lambda)

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