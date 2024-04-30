struct AffineCoupling{Model, Mask}
    nn::Model
    mask::Mask
end

Flux.@functor AffineCoupling

function (layer::AffineCoupling)(cnfg)
    x_pr = cnfg[1]
    logq_prec = cnfg[2]
    x_pr_frozen =  layer.mask .* x_pr
    x_pr_active = x_pr .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(x_pr_frozen, dims=ndims(x_pr_frozen)))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    fx = @. (1 - layer.mask) * t +  x_pr_active * exp(s) + x_pr_frozen
    logJ = sum((1 .- layer.mask) .* s, dims=1:(ndims(s)-1)) 
    return fx, logq_prec .- logJ
end
forward(layer::AffineCoupling, cnfg) = layer(cnfg)


function reverse(layer::AffineCoupling, fcnfg)
    fcnfg_frozen = fcnfg .* layer.mask
    fcnfg_active = fcnfg .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(fcnfg_frozen, dims=ndims(fcnfg_frozen)))
    s = nn_output[:,:,1,:] 
    t = nn_output[:,:,2,:] 
    x = @. (fcnfg_active - (1 - layer.mask) * t) * exp(-s) + fcnfg_frozen
    logJ = sum((1 .- layer.mask) .* (.-s), dims=1:(ndims(s)-1)) 
    return x, logJ
end

function freezing_mask(shape, parity)
    mask = ones(Float32, shape) .- Float32(parity)
    mask[1:2:end, 1:2:end] .= parity
    mask[2:2:end, 2:2:end] .= parity
    return mask
end

function create_affine_layers(hp::HyperParams)
    @timeit "Affine Layers" begin
        n_layers = hp.mp.n_layers
        device = hp.dp.device
        
        couplings = []
        for k in 0:(n_layers-1)
            net = build_cnn(hp.mp)
            mask = freezing_mask(hp.ap.lattice_shape, mod(k,2))
            tmp_coupling = AffineCoupling(Chain(net...), mask) 
            push!(couplings, tmp_coupling) 
        end
        affine_layers = Chain(couplings...) |> f32 |> device
        return affine_layers
    end
end

function evolve_prior_with_flow(prior, affine_layer; batchsize, lattice_shape, device, gauge_fix::Bool=true)
    @timeit "Evolve flow" begin    
        x_pr =  rand(prior, lattice_shape..., batchsize ) 
        if gauge_fix
            x_pr[1,1,:] .= 0.0 # gauge fixing
        end
        logq_prec = sum(logpdf.(prior, x_pr), dims=(1:ndims(x_pr)-1)) |> device
        x_pr_device = x_pr |> device
        xout, logq = affine_layer((x_pr_device, logq_prec ))
        return xout, logq
    end
end