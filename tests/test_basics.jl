using Revise
using LatFlow
using PyPlot, Statistics
using CircularArrays
##
# test action parameters

physpar = ActionParams(10, 1.0, 1.0)

# test Configurations
cnfg = Configuration{Float64}(physpar.dim)
cnfg.shape
cnfg.fields

# test sampling priors
prior = sampleNorm(1000)
mean(prior[1,1,:])

# test action
action = Phi4ScalarAction(1.0, 1.0)
eval_action = action(prior)

# test freezing_mask
mask =  freezing_mask(1, actionpar=physpar)

# test create conv net
net = create_conv_net(ModelParams())
Chain(net...)(Flux.unsqueeze(prior,3))

# test create affne layers
layer = create_affine_layers(ModelParams(), ActionParams(), DeviceParams(-1))
layer(Flux.unsqueeze(prior,3))

# test get training parameters

ps = get_training_param(layer)

## test evolve prior with flow

x, logq = evolve_prior_with_flow(layer, ActionParams(), TrainingParams(), DeviceParams(-1))
logp = -action(x)
compute_KL_div(-action(x), logq)


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

## test HyperParams
hp = HyperParams(DeviceParams(1), ActionParams(), ModelParams(), TrainingParams())
hp = HyperParams()

## test training
model, history = train(hp)

##
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