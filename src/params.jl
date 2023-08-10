@kwdef struct ActionParams
    L::Int64=8
    dim::Vector{Int64}=[L,L]
    m = 1.0
    lambda = 1.0
end
ActionParams(L, m, lambda) = ActionParams(L, [L, L], m, lambda)