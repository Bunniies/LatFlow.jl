module Models

using Flux
using Flux: onehotbatch, onecold, @epochs, @functor
using CUDA
using Distributions, TimerOutputs, Random, UnPack


using ..Params, ..Priors

include("convnet.jl")
export  build_cnn, get_training_param

include("affineCoupling.jl")
export AffineCoupling, forward, reverse, create_affine_layers, evolve_prior_with_flow, freezing_mask

end