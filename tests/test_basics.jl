using Revise
using LatFlow
using PyPlot, Statistics
using CircularArrays
##
# test action parameters

newpar = ActionParams(10, 1.0, 1.0)

# test Configurations
cnfg = Configuration{Float64}(newpar.dim)
cnfg.shape
cnfg.fields

# test action 
act = phi4ScalarAction(CircularArray(prior[:,:,1]))
for k in 1:100
    auxcnfg = Configuration{Float64}(newpar.dim)
    action = phi4ScalarAction(auxcnfg.fields)
    println("The phi4 action is: ", action)
    imshow(auxcnfg.fields)
    display(gcf())
    close("all")
end
phi4ScalarAction(prior[:,:,1])


# test new action
action = Phi4ScalarAction(1.0, 1.0)
aa = action(prior)
bb = sum(aa, dims=1:2)
cc  = dropdims(bb, dims=Tuple((1:2)))
# test sampling priors
prior = sampleNorm(1000)
mean(prior[1,4,:])