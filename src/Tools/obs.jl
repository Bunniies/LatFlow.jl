function green(cnfg::AbstractArray{T,N}, offsetX, id) where {T,N}
    @timeit "Green" begin
        shifts = broadcast(-, offsetX)
        batch_dim = N
        lattice_shape = (size(cnfg, i) for i in 1:N-1)
        cnfg_offset = circshift(cnfg, shifts)
        m_corr = Matrix{uwreal}(undef, Tuple(lattice_shape))
        m = Matrix{uwreal}(undef, Tuple(lattice_shape))
        m_offset = Matrix{uwreal}(undef, Tuple(lattice_shape))
        for t in 1:Tuple(lattice_shape)[1]
            for x in 1:Tuple(lattice_shape)[2]
                m_corr[t,x]   = uwreal(cnfg[t,x,:] .* cnfg_offset[t,x,:] .|> Float64, id)
                m[t,x]        = uwreal(cnfg[t,x,:] .|> Float64 , id)
                m_offset[t,x] = uwreal(cnfg_offset[t,x,:] .|> Float64, id)
            end
        end
        V = prod(lattice_shape)
        Gc = sum(m_corr .- m.*m_offset)/V
        return Gc
    end
end

function susceptibility(cfgs, id)
    @timeit "Susceptibility" begin
        lattice_shape = (cfgs |> size)[begin:end-1]
        acc = 0.0f0
        for s in collect(Iterators.product((1:l for l in lattice_shape)...))
            acc += green(cfgs, s, id)
        end
        return acc
    end
end