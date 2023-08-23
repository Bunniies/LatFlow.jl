module LatFlow

using Base:@kwdef
using Distributions, CircularArrays, PyPlot, Random, ProgressMeter, UnPack
using Flux
using Flux: onehotbatch, onecold, @epochs, @functor
using CUDA
using DataFrames

include("./params.jl")
include("./types.jl")
include("./tools.jl")
include("./couplingLayers.jl")
include("./training.jl")

export DeviceParams, ActionParams, ModelParams, TrainingParams, HyperParams
export Configuration, Phi4ScalarAction
export sampleNorm, freezing_mask, compute_KL_div, compute_ESS, get_training_param
export AffineCoupling, create_conv_net, create_affine_layers, evolve_prior_with_flow
export train
end 