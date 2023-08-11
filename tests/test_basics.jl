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