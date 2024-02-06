function train(hp::HyperParams, action, prior; flog::Union{String, IOStream}="", savemode::Bool=true, nfws::NFworkspace)

    @unpack dp, ap, mp, tp = hp
    @unpack iterations, epochs, batch_size, eta_lr = tp
    device = dp.device

    affine_layers = create_affine_layers(hp, nfws=nfws)
    ps = get_training_param(affine_layers)
    opt = Adam(tp.eta_lr)

    history = DataFrame(
        "epochs"          => Int[],
        "loss"            => Float64[],
        "ess"             => Float64[],
        "acceptance_rate" => Float64[]
    )

    best_ess = 0.0
    best_ess_epoch = 1
    best_acc = 0.0
    best_acc_epoch = 1
    @showprogress for epoch in 1:epochs
        println(flog, "Epoch: $(epoch)")
        
        # train mode 
        Flux.trainmode!(affine_layers)
        @timeit "Training" begin
            for _ in 1:iterations
                x_pr = rand(prior, ap.lattice_shape..., batch_size ) |> device
                logq_prec = sum(logpdf.(prior, x_pr), dims=1:ndims(x_pr)-1) |> device
                #x_pr_dev = x_pr |> device

                grads = Flux.gradient(ps) do 
                    x_out, logq_ = affine_layers((x_pr, logq_prec)) 
                    logq = dropdims(logq_, dims=(1,ndims(logq_)-1))
                    logp = - action(x_out)
                    loss = compute_KL_div(logp, logq |> device)
                end
                Flux.Optimise.update!(opt, ps, grads)
            end
        end 

        # test mode 
        Flux.testmode!(affine_layers)
        x_out, logq = evolve_prior_with_flow(prior, affine_layers, batchsize=batch_size, lattice_shape=ap.lattice_shape, device=device)
        logq = dropdims(logq, dims=(1,ndims(logq)-1))

        logp = -action(x_out)
        loss = compute_KL_div(logp, logq |> device)
        ess = compute_ESS(logp, logq |> device)
        println(flog, "    loss: $(loss)")
        println(flog, "    ess:  $(ess)")

        nsamples=8192
        hist_mcmc = build_mcmc(prior, affine_layers, action, batchsize=batch_size, nsamples=nsamples, lattice_shape=ap.lattice_shape, device=device )
        acc = hist_mcmc[!,"accepted"] |> mean
        println(flog, "    acc: $(acc)")
        push!(history[!,"epochs"], epoch)
        push!(history[!,"loss"], loss)
        push!(history[!,"ess"], ess)
        push!(history[!,"acceptance_rate"], acc)

        if savemode
            # save model with best ess
            if ess >= best_ess
                println(flog, "    new best ess at epoch $(epoch)")
                best_ess = ess
                best_ess_epoch = epoch
                BSON.@save joinpath("trainedNet", "model_best_ess.bson") affine_layers
                BSON.@save joinpath("trainedNet", "history_best_ess.bson") hist_mcmc
            end
            # save model with best acc
            if acc >= best_acc
                println(flog, "    new best acceptance rate at epoch $(epoch)")
                best_acc = acc
                best_acc_epoch = epoch
                BSON.@save joinpath("trainedNet", "model_best_acc.bson") affine_layers
                BSON.@save joinpath("trainedNet", "history_best_acc.bson") hist_mcmc
            end
        end
            
        if acc >= 0.7
            break
        end

    end

    # logging
    println(flog, " ")
    println(flog, history)
    println(flog, " ")

    return affine_layers, history
    # return history
end