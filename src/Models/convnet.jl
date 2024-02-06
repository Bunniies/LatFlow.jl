function build_cnn(mp::ModelParams)  
    ch_tot = [mp.inCh, mp.hidden_ch..., mp.outCh]
    kernel_size = tuple(fill(mp.kernel_size,2)...)

    net = []
    for kch in eachindex(ch_tot[1:end-2])
        if !mp.use_bn
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], leakyrelu, pad=0, stride=1))
        else
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch_tot[kch]=>ch_tot[kch+1], pad=0, stride=1))
            push!(net, BatchNorm(ch_tot[kch+1], leakyrelu))
        end
    end
    
    if mp.use_tanh_last
        ch = ch_tot[end-1]
        ch_last = ch_tot[end]
        if !mp.use_bn
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Conv(kernel_size, ch=>ch_last, tanh, pad=0, stride=1))
        else
            push!(net, x-> Flux.NNlib.pad_circular(x,(1,1,1,1) ))
            push!(net, Chain(Conv(kernel_size, ch=>ch_last, pad=0, stride=1), BatchNorm(ch_last, tanh)))
        end
    end
    return net
end

function get_training_param(layer)
    ps = Flux.params(layer)
    for k in eachindex(layer)
        delete!(ps, layer[k].mask)
        delete!(ps,layer[k].nfws)
    end
    return ps
end