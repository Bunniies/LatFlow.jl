module LatFlow

include("Params/Params.jl")
using .Params
export DeviceParams, ModelParams, TrainingParams, HyperParams
export ActionParams

include("Priors/Priors.jl")
using .Priors
export get_prior

include("Actions/Actions.jl")
using .Actions
export Phi4ScalarAction


include("Models/Models.jl")
using .Models
export build_cnn, get_training_param
export AffineCoupling, forward, reverse, create_affine_layers, evolve_prior_with_flow, freezing_mask

include("Tools/Tools.jl")
using .Tools
export compute_ESS, compute_KL_div
export build_mcmc
export green, susceptibility

include("Train/Train.jl")
using .Train
export train


# using Distributions, CircularArrays, PyPlot, Random, ProgressMeter, UnPack
# using Flux
# using Flux: onehotbatch, onecold, @epochs, @functor
# using CUDA



end 