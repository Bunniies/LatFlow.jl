function train!(affine_layers, hp::HyperParams, action, prior; flog::Union{String, IOStream}="")

    println(flog, "#==============================#")
    println(flog, "#======= START TRAINING =======#")
    println(flog, "#==============================#")

    @unpack dp, ap, mp, tp = hp
    @unpack iterations, epochs, batch_size, eta_lr = tp
    device = dp.device

    # affine_layers = create_affine_layers(hp)
    ps = get_training_param(affine_layers)

    opt = Adam(tp.eta_lr)

    history = DataFrame(
        "epochs"          => Int[],
        "loss"            => Float64[],
        "ess"             => Float64[],
        "timing"          => Float64[],
        "acceptance_rate" => Float64[]
    )

    @showprogress for epoch in 1:epochs
        println(flog, "Epoch: $(epoch)")
        
        # train mode 
        Flux.trainmode!(affine_layers)
        ts = @timed begin
            for _ in 1:iterations

                x_pr = rand(prior, ap.lattice_shape..., batch_size ) 
                logq_prec = sum(logpdf.(prior, x_pr), dims=1:ndims(x_pr)-1) |> device
                x_pr_dev = x_pr |> device

                grads = Flux.gradient(ps) do 
                    x_out, logq_ = affine_layers((x_pr_dev, logq_prec)) 
                    logq = vcat(logq_)
                    logp = - action(x_out)
                    loss = compute_KL_div(logp, logq |> device)
                end
                Flux.Optimise.update!(opt, ps, grads)
            end
        end 

        # test mode 
        Flux.testmode!(affine_layers)
        x_out, logq = evolve_prior_with_flow(prior, affine_layers, batchsize=batch_size, lattice_shape=ap.lattice_shape, device=device)
        #logq = dropdims(logq, dims=(1,ndims(logq)-1))
        logq = vcat(logq...)

        logp = -action(x_out)
        loss = compute_KL_div(logp, logq |> device)
        ess = compute_ESS(logp, logq |> device)
        println(flog, "    loss: $(loss)")
        println(flog, "    ess:  $(ess)")

        push!(history[!,"epochs"], epoch)
        push!(history[!,"timing"], ts.time)
        push!(history[!,"loss"], loss)
        push!(history[!,"ess"], ess)
        push!(history[!,"acceptance_rate"], 0.0)
    end
    # return affine_layers, history
    return history
end