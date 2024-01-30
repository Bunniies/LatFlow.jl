using Revise
using LatFlow
using PyPlot, Statistics
using Flux

physpar = ActionParams(10, 1.0, 1.0)
prior = sampleNorm(1000, actionpar=physpar)
mask = freezing_mask(1, actionpar=physpar)

cnfg_frozen =  prior .* mask
Flux.unsqueeze(cnfg_frozen, dims=ndims(cnfg_frozen))