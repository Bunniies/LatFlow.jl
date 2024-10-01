function train(hp::HyperParams, action, prior; flog::Union{String, IOStream}="", savemode::Bool=true, gauge_fix::Bool=true, load_model::Union{String, Nothing}=nothing )

    @unpack dp, ap, mp, tp = hp
    @unpack iterations, epochs, batch_size, eta_lr = tp
    device = dp.device

    if isnothing(load_model)
        affine_layers = create_affine_layers(hp)
        ps = get_training_param(affine_layers)
    else
        println(flog, "model loaded from: ", load_model)
        model_state = JLD2.load(load_model, "model_state")
        affine_layers = create_affine_layers(hp)   
        Flux.loadmodel!(affine_layers, model_state)
        ps = get_training_param(affine_layers)
    end

    # opt = Flux.Optimise.Optimiser(Flux.Optimise.ClipNorm(0.5f0), Flux.Adam(tp.eta_lr))
    opt = Flux.Optimise.Optimiser(Flux.Adam(tp.eta_lr))
    sched = Stateful(Step(tp.eta_lr, 0.9, 1000))

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
                if gauge_fix
                    x_pr[1,1,:] .= 0.0f0 
                end
                logq_prec = sum(logpdf.(prior, x_pr), dims=1:ndims(x_pr)-1) |> device

                grads = Flux.gradient(ps) do 
                    x_out, logq_ = affine_layers((x_pr, logq_prec)) 
                    logq = dropdims(logq_, dims=(1,ndims(logq_)-1))
                    logp = - action(x_out)
                    loss = compute_KL_div(logp, logq |> device)
                end
                Flux.update!(opt, ps, grads)
                opt.os[1].eta = next!(sched)     
            end
        end 

        # test mode 
        Flux.testmode!(affine_layers)
        x_out, logq = evolve_prior_with_flow(prior, affine_layers, batchsize=batch_size, lattice_shape=ap.lattice_shape, device=device, gauge_fix=gauge_fix)
        logq = dropdims(logq, dims=(1,ndims(logq)-1))

        logp = -action(x_out)
        loss = compute_KL_div(logp, logq |> device)
        ess = compute_ESS(logp, logq |> device)
        println(flog, "    loss: $(loss)")
        println(flog, "    ess:  $(ess)")
        println(flog, "    eta:  $(opt.os[1].eta)")
        println(
        "    loss: $(loss)")

        # nsamples=8192
        # hist_mcmc = build_mcmc(prior, affine_layers, action, batchsize=batch_size, nsamples=nsamples, lattice_shape=ap.lattice_shape, device=device, gauge_fix=gauge_fix)
        nsamples=2^8
        hist_mcmc = build_mcmc(prior, affine_layers, action, batchsize=2^5, nsamples=nsamples, lattice_shape=ap.lattice_shape, device=device, gauge_fix=gauge_fix)
        acc = hist_mcmc[!,"accepted"] |> mean
        println(flog, "     acc: $(acc)")
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
                jldsave(joinpath("trainedNet", "model_best_ess_ep$(epoch).jld2"), model_state=Flux.state(cpu(affine_layers)))
                BSON.@save joinpath("trainedNet", "history_best_ess_ep$(epoch).bson") hist_mcmc
            end
            # save model with best acc
            if acc >= best_acc
                println(flog, "    new best acceptance rate at epoch $(epoch)")
                best_acc = acc
                best_acc_epoch = epoch
                # BSON.@save joinpath("trainedNet", "model_best_acc_ep$(epoch).bson") affine_layers
                jldsave(joinpath("trainedNet","model_best_acc_ep$(epoch).jld2"), model_state=Flux.state(cpu(affine_layers)))
                BSON.@save joinpath("trainedNet", "history_best_acc_ep$(epoch).bson") hist_mcmc
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