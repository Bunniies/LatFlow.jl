function build_mcmc(prior, layer, action; batchsize, nsamples, lattice_shape, device=cpu, seed=3430, gauge_fix::Bool=true)

    rng = MersenneTwister(seed)
    mcmc_hist = DataFrame(
        "logp"     => Float32[],
        "logq"     => Float32[],
        "config"   => Array{Float32, 2}[],
        "accepted" => Bool[],
    )

    counter = 0
    @timeit "MCMC step" begin
        
        for _ in 1:round(Int, nsamples/batchsize)
            x_out_, logq = evolve_prior_with_flow(prior, layer, batchsize=batchsize, lattice_shape=lattice_shape, device=device, gauge_fix=gauge_fix)
            logq = vec(logq) |> cpu
            logp = -action(x_out_) |> cpu
            x_out = x_out_ |> cpu

            for k in 1:batchsize
                x_proposed = x_out[:,:,k]
                logq_proposed = logq[k]
                logp_proposed = logp[k]

                if isempty(mcmc_hist[!,"logp"])
                    acc = true
                else
                    logq_prev = mcmc_hist[!, "logq"][end]
                    logp_prev = mcmc_hist[!, "logp"][end]
                    p_acc = exp((logp_proposed - logq_proposed) - (logp_prev - logq_prev))
                    p_acc = min(one(eltype(p_acc)), p_acc)
                    coin = rand(rng, eltype(p_acc))
                    if coin < p_acc
                        acc = true
                    else
                        acc = false
                        x_proposed = mcmc_hist[!, "config"][end]
                        logq_proposed = logq_prev
                        logp_proposed = logp_prev
                    end
                end
                push!(mcmc_hist[!, "logp"], logp_proposed)
                push!(mcmc_hist[!, "logq"], logq_proposed)
                push!(mcmc_hist[!, "config"], x_proposed)
                push!(mcmc_hist[!, "accepted"], acc)
            end
            counter += batchsize
            if counter >= nsamples
                break
            end
        end
    end
    return mcmc_hist
end