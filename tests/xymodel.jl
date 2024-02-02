import Pkg
using Revise
Pkg.activate("/Users/alessandroconigli/.julia/dev/LatFlow")
Pkg.instantiate()

using CUDA, ArgParse, TOML, Logging, ADerrors, Flux
using LatFlow, PyPlot, Statistics, TimerOutputs
using Random
using Distributions

lat_shape = (8,8)
beta= 2.f0

ap = ActionParams(lattice_shape=lat_shape, beta=beta)
dp = DeviceParams(0)
tp = TrainingParams(iterations=200, epochs=100, batch_size=64)
mp = ModelParams(n_layers=4, hidden_ch=[4,4] )
hp = HyperParams(dp, ap, mp, tp)

prior = get_prior("Normal", mu=0., k=4. *beta, a=0, b=6.28)
action = XYmodelAction(beta)

rand(prior, (8,8))
layers, train_hist = train(hp, action, prior, savemode=false)

## check trained configs
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
#savefig("log/trained_config.pdf")
close("all")

##

println("MCMC step")
nsamples= 11000
ntherm = 1000
mcmc_hist = build_mcmc(prior, layers, action, batchsize=hp.tp.batch_size, nsamples=nsamples, lattice_shape=hp.ap.lattice_shape, device=hp.dp.device )
cnfg = Flux.MLUtils.batch(mcmc_hist[!, "config"][ntherm:end])
acceptance = mcmc_hist[!, "accepted"] |> mean
susc = susceptibility(cnfg); uwerr(susc)
susc