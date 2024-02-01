struct NFworkspace{T}
    xpr
    xpr4d 
    function NFworkspace(::Type{T}, hp::HyperParams) where {T <: AbstractFloat}
        @timeit "Allocating NFWorkspace" begin
            @unpack dp, ap, mp, tp = hp

            if dp.device == Flux.cpu
                _xpr = Array{T,3}(undef, ap.lattice_shape..., tp.batch_size)
                _xpr4d = Array{T,4}(undef, ap.lattice_shape..., 1, tp.batch_size)
            elseif dp.device == Flux.gpu
                _xpr = CuArray{T,3}(undef, ap.lattice_shape..., tp.batch_size) 
                _xpr4d = CuArray{T,4}(undef, ap.lattice_shape..., 1, tp.batch_size)
            end

            return new{T}(_xpr, _xpr4d)
        end
    end
    
end