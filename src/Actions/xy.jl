struct XYmodelAction
    beta::Float32
end

function (action::XYmodelAction)(cnfg)
    dim = ndims(cnfg)
    action_density = sum( cos.(cnfg - circshift(cnfg, Flux.onehot(k,1:dim))) for k in 1:dim-1 )
    act = sum(action_density, dims=1:dim-1)
    return  -1 .* action.beta  .* dropdims(act, dims=Tuple((1:dim-1))) 
end