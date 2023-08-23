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

function compute_KL_div(logp, logq)
    return mean(logq .- logp)
end

function compute_ESS(logp, logq)
    log_diff = logp - logq
    log_ess = 2*logsumexp(log_diff) - logsumexp(2*log_diff)
    return exp(log_ess) / length(log_diff)
end

function get_training_param(layer)
    ps = Flux.params(layer)
    for k in eachindex(layer)
        delete!(ps, layer[k].mask)
    end
    return ps
end