struct Phi4ScalarAction
    m2::Float32
    lambda::Float32
end

function (action::Phi4ScalarAction)(cnfg)
    U = cnfg .^2 .* action.m2 .+ cnfg .^4 .* action.lambda
    dim = ndims(cnfg) 
    kin = sum(cnfg .* ( 2 * cnfg - circshift(cnfg, -Flux.onehot(k,1:dim)) - circshift(cnfg, Flux.onehot(k,1:dim)) ) for k in 1:dim-1)
    action_sum = sum(kin .+ U, dims=1:dim-1)

    return dropdims(action_sum, dims=Tuple((1:dim-1)))
end