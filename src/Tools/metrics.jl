function compute_KL_div(logp, logq)
    return mean(logq .- logp)
end

function compute_ESS(logp, logq)
    log_diff = logp - logq
    log_ess = 2*logsumexp(log_diff) - logsumexp(2*log_diff)
    return exp(log_ess) / length(log_diff)
end