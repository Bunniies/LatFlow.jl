struct NFworkspace{T}
    xpr
    function NFworkspace(::Type{T}, hp::HyperParams) where {T <: AbstractFloat}
        @timeit "Allocating NFWorkspace" begin
            @unpack dp, ap, mp, tp = hp

            if dp.device == Flux.cpu
                _xpr = Array{T,3}(undef, ap.lattice_shape..., tp.batch_size)
            elseif dp.device == Flux.gpu
                _xpr = CuArray{T,3}(undef, ap.lattice_shape..., tp.batch_size) 
            end

            return new{T}(_xpr)
        end
    end
    
end