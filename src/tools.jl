function sampleNorm(N; actionpar::ActionParams=ActionParams())
    prior = rand(Normal{Float32}(0.f0, 1.f0), actionpar.dim..., N)
    return prior
end


function freezing_mask(parity::Int64; actionpar::ActionParams=ActionParams())
    mask = ones(actionpar.dim...) .- parity
    mask[1:2:end, 1:2:end] .= parity
    mask[2:2:end, 2:2:end] .= parity
    return mask
end