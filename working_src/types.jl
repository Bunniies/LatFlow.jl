struct Configuration{T}
    shape::Vector{Int64}
    fields::CircularArray{T}
    function Configuration{T}(lat_size::Vector{Int64}) where T
        _fields = CircularArray(randn(T,tuple(lat_size...))) 
        return new(lat_size, _fields)
    end
    function Configuration{T}(lat_size::Vector{Int64}, _fields::CircularArray{T}) where T
        return new(lat_size, _fields)
    end
end

struct Phi4ScalarAction
    m::Float32
    lambda::Float32
end

function (action::Phi4ScalarAction)(cnfg)
    U = cnfg .^2 .* action.m^2 .+ cnfg .^4 .* action.lambda
    dim = ndims(cnfg) 
    kin = zeros(size(cnfg))
    for k in 1:(dim-1)
        kin .+= cnfg .* ( 2 * cnfg - circshift(cnfg, -Flux.onehot(k,1:dim)) - circshift(cnfg, Flux.onehot(k,1:dim)) )
    end
    action_sum = sum(kin .+ U, dims=1:dim-1)
    return dropdims(action_sum, dims=Tuple((1:dim-1)))
end

