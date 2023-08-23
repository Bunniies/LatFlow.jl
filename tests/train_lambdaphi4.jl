using Revise
using LatFlow
using PyPlot, Statistics
##

mp = ModelParams()
tp = TrainingParams()
ap = ActionParams(8, -4.0, 8.0)
dp = DeviceParams(-1)

hp = HyperParams(dp, ap, mp, tp)

##
model, history = train(hp)


## plot cnfgs
x, logq = evolve_prior_with_flow(model, ActionParams(), TrainingParams(), DeviceParams(-1))

fig, ax = plt.subplots(5,5, dpi=125, figsize=(5,5))
for i in 1:5
    for j in 1:5
        ind = i*5 + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end
display(gcf())
close("all")