module LatFlow

using Base:@kwdef
using Distributions, CircularArrays, PyPlot, Random, ProgressMeter
using Flux
using Flux: onehotbatch, onecold, @epochs, @functor
using CUDA

include("./params.jl")
include("./types.jl")
include("./tools.jl")
include("./couplingLayers.jl")

export DeviceParams, ActionParams, ModelParams, TrainingParams
export Configuration, Phi4ScalarAction
export sampleNorm, freezing_mask
export AffineCoupling, create_conv_net, create_affine_layers, evolve_prior_with_flow

end 