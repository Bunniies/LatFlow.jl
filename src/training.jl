function train(hp::HyperParams)
    dp = hp.dp
    ap = hp.ap
    mp = hp.mp
    tp = hp.tp
    @info "Model setup"
    @unpack iterations, epochs, batch_size, eta_lr = hp.tp
    device = hp.dp.device

    nn_model = create_affine_layers(hp.mp, hp.ap, hp.dp)
    ps = get_training_param(nn_model)
    opt = Adam(hp.tp.eta_lr)
    action = Phi4ScalarAction(hp.ap.m, hp.ap.lambda)

    history = DataFrame(
        "epochs"          => Int[],
        "loss"            => Float64[],
        "ess"             => Float64[],
        "timing"          => Float64[],
        "acceptance_rate" => Float64[]
    )

    @info "Training model"
    @showprogress for epoch in 1:epochs
        @info "epoch=$epoch"
        
        # train mode 
        Flux.trainmode!(nn_model)
        ts = @timed begin
            for _ in 1:iterations
                prior = sampleNorm(hp.tp.batch_size) |> device
                # logq = sum(logpdf.(Normal{Float32}(0.f0,1.f0), prior), dims=1:ndims(prior)-1) |> device
                x_out, logq = evolve_prior_with_flow(nn_model, hp.ap, hp.tp, hp.dp)
                logp = - action(x_out)
                grads = Flux.gradient(ps) do 
                    loss = compute_KL_div(logp, logq)
                end
                Flux.Optimise.update!(opt, ps, grads)
            end
        end 

        # test mode 
        Flux.testmode!(nn_model)
        prior = sampleNorm(hp.tp.batch_size) |> device
        x_out, logq = evolve_prior_with_flow(nn_model, hp.ap, hp.tp, hp.dp)
        logq = dropdims(logq, dims=(1,ndims(logq)-1))
        logp = -action(x_out)
        loss = compute_KL_div(logp, logq)
        @show loss
        ess = compute_ESS(logp, logq)
        @show ess 
        push!(history[!,"epochs"], epoch)
        push!(history[!,"timing"], ts.time)
        push!(history[!,"loss"], loss)
        push!(history[!,"ess"], ess)
        push!(history[!,"acceptance_rate"], 0.0)
    end
    return nn_model, history
end