struct AffineCoupling{Model, Mask, Workspace}
    nn::Model
    mask::Mask
    nfws::Workspace
end

Flux.@functor AffineCoupling

function (layer::AffineCoupling)((x_pr, logq_prec))
    
    @unpack nn, mask, nfws = layer            
    nfws.xfrozen .= mask .* x_pr
    nfws.xactive .= x_pr .* (1 .- mask)
    nfws.xpr4d .= Flux.unsqueeze(nfws.xfrozen, dims=ndims(nfws.xfrozen))
    @timeit "model" begin
        nfws.nnoutput .= nn(nfws.xpr4d)
    end
    nfws.s .= nfws.nnoutput[:,:,1,:] 
    nfws.t .= nfws.nnoutput[:,:,2,:] 
    fx = @. (1 - mask) * nfws.t + nfws.xactive * exp(nfws.s) + nfws.xfrozen
    logJ = sum((1 .- mask) .* nfws.s, dims=1:(ndims(nfws.s)-1)) 
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

function create_affine_layers(hp::HyperParams; nfws::NFworkspace)
    @timeit "Affine Layers" begin
        n_layers = hp.mp.n_layers
        device = hp.dp.device
        
        couplings = []
        for k in 0:(n_layers-1)
            net = build_cnn(hp.mp)
            mask = freezing_mask(hp.ap.lattice_shape, mod(k,2))
            tmp_coupling = AffineCoupling(Chain(net...), mask, nfws) 
            push!(couplings, tmp_coupling) 
        end
        affine_layers = Chain(couplings...) |> f32 |> device
        return affine_layers
    end
end

function evolve_prior_with_flow(prior, affine_layer; device, nfws::NFworkspace)
    @timeit "Evolve flow" begin    
        rand!(prior, nfws.xpr3d ) 
        logq_prec = sum(logpdf.(prior, nfws.xpr3d), dims=(1:ndims(nfws.xpr3d)-1)) 
        xout, logq = affine_layer((nfws.xpr3d, logq_prec ))
        return xout, logq
    end
end