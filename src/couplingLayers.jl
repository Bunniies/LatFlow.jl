struct AffineCoupling{Model, Mask}
    nn::Model
    mask::Mask
end

@functor AffineCoupling

function forward(layer::AffineCoupling, cnfg)
    cnfg_frozen = cnfg .* layer.mask
    cnfg_active = cnfg .* (1 .- layer.mask)
    nn_output = layer.nn(Flux.unsqueeze(cnfg_frozen, 3))
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
    return x, logJ
end

function create_conv_net(mp::ModelParams)  
    seed = mp.seed
    ch_tot = [mp.inCh, mp.hidden_ch..., mp.outCh]
    kernel_size = tuple(fill(mp.kernel_size,2)...)

    net = []
    for kch in eachindex(ch_tot[1:end-1])
        aux_net = Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], pad=1, stride=1)
        if !mp.use_bn
            push!(net, Chain(aux_net, x -> leakyrelu.(x)) )
        else
            push!(net, Chain(aux_net, BatchNorm(ch_tot[kch+1], leakyrelu)))
        end
    end
    
    if mp.use_tanh_last
        ch = ch_tot[end-1]
        ch_last = ch_tot[end]
        if !mp.use_bn
            net[end] = Conv(kernel_size, ch=> ch_last, tanh, pad=1, stride=1)
        else
            net[end] = Chain(Conv(kernel_size, ch=> ch_last, pad=1, stride=1), BatchNorm(ch_last, tanh))
        end
    end
    return net
end

function create_affine_layers(mp::ModelParams, ap::ActionParams, dp::DeviceParams)
    n_layers = mp.n_layers
    device = dp.device
    
    couplings = []
    for k in 1:n_layers
        net = create_conv_net(mp)
        mask = freezing_mask(mod(k,2), actionpar=ap)
        tmp_coupling = AffineCoupling(Chain(net...), mask) 
        push!(couplings, tmp_coupling) 
    end
    affine_layers = Chain(couplings...) |> f32 |> device
end

function evolve_prior_with_flow(layer, ap::ActionParams, tp::TrainingParams, dp::DeviceParams)
    batch_size = tp.batch_size
    x_pr = sampleNorm(batch_size, actionpar=ap)
    logq = sum(logpdf.(Normal{Float32}(0.f0,1.f0), x_pr), dims=(1:ndims(x_pr)-1)) |> dp.device
    for item in layer
        x_pr, logJ = forward(item, x_pr)
        logq = logq - logJ
    end
    return x_pr, logq
end