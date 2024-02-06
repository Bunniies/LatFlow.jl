struct NFworkspace{T,L,B}
    # T: type
    # L: lattice size 
    # B: batch size
    latdim
    xpr3d
    xpr4d
    xactive
    xfrozen 
    nnoutput
    s
    t
    function NFworkspace{L,B}(::Type{T}, hp::HyperParams) where {T <: AbstractFloat, L, B}
        @timeit "Allocating NFWorkspace" begin
            @unpack dp, ap, mp, tp = hp
            latdim = (L,L)
    
            if dp.device == Flux.cpu
                _xpr = Array{T,3}(undef, latdim..., B )
                _xpr4d = Array{T,4}(undef, latdim..., 1, B)
                _xactive = Array{T,3}(undef, latdim..., B)
                _xfrozen = Array{T,3}(undef, latdim..., B)
                _nnoutput = Array{T,4}(undef, latdim..., 2, B)
                _s = Array{T,3}(undef, latdim..., B)
                _t = Array{T,3}(undef, latdim..., B)

            elseif dp.device == Flux.gpu
                _xpr = CuArray{T,3}(undef, latdim..., B) 
                _xpr4d = CuArray{T,4}(undef, latdim..., 1, B)
                _xactive = CuArray{T,3}(undef, latdim..., B)
                _xfrozen = CuArray{T,3}(undef, latdim..., B)
                _nnoutput = CuArray{T,4}(undef, latdim..., 2, B)
                _s = CuArray{T,3}(undef, latdim..., B)
                _t = CuArray{T,3}(undef, latdim..., B)
            end

            return new{T,L,B}(latdim, _xpr, _xpr4d, _xactive, _xfrozen, _nnoutput, _s, _t)
        end
    end
    
end