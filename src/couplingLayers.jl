struct AffineCoupling{Model, Mask}
    nn::Model
    mask::Mask
end

@functor AffineCoupling

function forward(layer::AffineCoupling, cnfg)
    cnfg_frozen = cnfg .* layer.mask
    cnfg_active = cnfg .* (1 .- layer.mask)
    nn_output = layer.net(Flux.unsqueeze(cnfg_frozen, 3))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    fx = @. (1 - layer.mask) * t +  cnfg_active * exp(s) + cnfg_frozen
    logJ = sum((1 .- layer.mask) .* s, dims=1:(ndims(s)-1)) 
    return fx, logJ
end

function reverse(layer::AffineCoupling, fcnfg)
    fcnfg_frozen = fcnfg .* layer.mask
    fcnfg_active = fcnfg .* (1 .- layer.mask)
    nn_output = layer.net(Flux.unsqueeze(fcnfg_frozen, 3))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    x = @. (fcnfg_active - (1 - layer.mask) * t) * exp(-s) + fcnfg_frozen
    logJ = sum((1 .- layer.mask) .* (.-s), dims=1:(ndims(s)-1)) 
    return fx, logJ
end