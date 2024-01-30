import Pkg

Pkg.activate("/Users/alessandroconigli/.julia/dev/LatFlow")
Pkg.instantiate()

using CUDA, ArgParse, TOML, Logging, ADerrors, Flux
using LatFlow, PyPlot, Statistics



function parse_flag()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-i"
        help = "input file"
        required = true
        arg_type = String
    end
    return parse_args(s)
end

function parse_arg(fname)

    f = TOML.parsefile(fname)

    ap = ActionParams(lattice_shape=Tuple(f["Action"]["lattice_shape"]), m2=Float32(f["Action"]["m2"]), lambda=Float32(f["Action"]["lambda"]), beta=Float32(f["Action"]["beta"]))
    dp = DeviceParams(f["Device"]["id"])
    tp = TrainingParams(f["Training"]["iterations"], f["Training"]["epochs"], f["Training"]["batch_size"], f["Training"]["eta_lr"])
    fmod = f["Model"]
    mp = ModelParams(fmod["seed"], fmod["inCh"], fmod["outCh"], fmod["n_layers"], fmod["hidden_ch"], fmod["kernel_size"], fmod["use_tanh"], fmod["use_bn"])
    hp = HyperParams(dp, ap, mp, tp)

    if f["Action"]["name"] == "phi4"
        filename = string("./", f["Run"]["log_dir"], f["Run"]["name"], "_m2", f["Action"]["m2"], "_lamda", f["Action"]["lambda"], ".log")
    elseif  f["Action"]["name"] == "xy"
        filename = string("./", f["Run"]["log_dir"], f["Run"]["name"], "_beta", f["Action"]["beta"], ".log")
    end
    
    flog = open(filename, "w+")
    
    println(flog, "# [Environment Info]")
    println(flog, "# User:       ", f["Run"]["user"])
    println(flog, "# Host:       ", f["Run"]["host"])
    println(flog, "# Run name:   ", f["Run"]["name"])
    println(flog, "# Load model: ", f["Run"]["pretrained"])
    println(flog, " ")

    println(flog, "# [Device Info]")
    println(flog, "# Device:    ", string(dp.device))
    println(flog, "# Device id: ", dp.id)
    println(flog, " ")

    println(flog, "# [Lattice Setup]")
    println(flog, "# Action:       ", f["Action"]["name"])
    println(flog, "# Lattice size: ", f["Action"]["lattice_shape"])
    if f["Action"]["name"] == "phi4"
        println(flog, "# m2:           ", f["Action"]["m2"])
        println(flog, "# lambda:       ", f["Action"]["lambda"])
    elseif  f["Action"]["name"] == "xy"
        println(flog, "# beta:         ", f["Action"]["beta"])
    end
    println(flog, " ")

    println(flog, "# [Model Parameters]")
    println(flog, "# Input channel:   ", mp.inCh)
    println(flog, "# Output channel:  ", mp.outCh)
    println(flog, "# Affine layers:   ", mp.n_layers)
    println(flog, "# Hidden channels: ", mp.hidden_ch)
    println(flog, "# Kernel size:     ", mp.kernel_size)
    println(flog, "# Use tanh:        ", mp.use_tanh_last)
    println(flog, "# Batch norm:      ", mp.use_bn)
    println(flog, "# Seed:            ", mp.seed)
    println(flog, " ")

    println(flog, "# [Training Parameters]")
    println(flog, "# Epochs:        ", tp.epochs)
    println(flog, "# Iterations:    ", tp.iterations)
    println(flog, "# Batch size     ", tp.batch_size)
    println(flog, "# Learning rate: ", tp.eta_lr)
    println(flog, " ")

    println(flog, "# [Priors]")
    println(flog, "Prior: ", f["Priors"]["prior"])
    println(flog, " ")

    nsamples = f["Mcmc"]["nsamples"]
    ntherm = f["Mcmc"]["ntherm"]
    println(flog, "# [Mcmc]")
    println(flog, "Nsamples: ", nsamples )
    println(flog, "Ntherm:   ", ntherm)

    # println(flog, Pkg.status())

    println(flog, " ")
    try
        println(flog, CUDA.versioninfo())
    catch
        println(flog, "# NVIDIA driver for GPU not installed. Training is performed on CPU.")
    end
    println(flog, " ")
    println(flog, "# END [Environment Info]")
    println(flog, "#============================================================================#")
    println(flog, " ")


    prior = get_prior(f["Priors"]["prior"])
    action = if f["Action"]["name"] == "phi4"
                Phi4ScalarAction(ap.m2, ap.lambda)
            elseif f["Action"]["name"] == "xy"
                error("Action for xy model not yet implemented.")
            end

    layers = if !f["Run"]["pretrained"]
                layers = create_affine_layers(hp)
            else
                error("Load pretrained model not yet implemented.")
            end

    return hp, prior, action, layers, flog, nsamples, ntherm, f["Model"]["seed"]
end


#============= SIMULATION ===========#

parsed_flags = parse_flag()
infile = parsed_flags["i"]
hp, prior, action, layers, flog, nsamples, ntherm, seed = parse_arg(infile)

# training
println(flog, "# [Training]")
train_hist = train!(layers, hp, action, prior, flog=flog)

println(flog, " ")
println(flog, train_hist)
println(flog, " ")
println(flog, "# Total training time: ", sum(train_hist[!, "timing"]))
println(flog, "# END [Training]")
println(flog, " ")



# check trained configs
x, logq = evolve_prior_with_flow(prior, layers, batchsize=hp.tp.batch_size, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
x = x |> cpu

ig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end

display(gcf())
savefig("log/trained_config.pdf")
close("all")

# check trained action
x, logq = evolve_prior_with_flow(prior, layers, batchsize=1024, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device)
x = cpu(x)
S_eff = -logq |> cpu
action = Phi4ScalarAction(hp.ap.m2, hp.ap.lambda)

S = action(x)
fit_b = mean(S) - mean(S_eff)
@show fit_b
#print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20,
    #range=[[5, 35], [-5, 25]]
)

xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
display(gcf())
savefig("log/trained_action.pdf")
close("all")

# MCMC step

println(flog, "# [MCMC steps]")
mcmc_hist = build_mcmc(prior, layers, action, batchsize=hp.tp.batch_size, nsamples=nsamples, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device, seed=seed)
cnfg = Flux.MLUtils.batch(mcmc_hist[!, "config"][ntherm:end])
acceptance = mcmc_hist[!, "accepted"] |> mean
println(flog, "# Acceptance Rate: ", acceptance)
println(flog, "# END [MCMC steps]")
println(flog, " ")


# Observable measurements

susc = susceptibility(cnfg); uwerr(susc)
println(flog,"# Susceptibility: ", susc )









