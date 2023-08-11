module LatFlow

using Distributions, CircularArrays, PyPlot, Random
using Flux
using Flux: onehotbatch, onecold, @epochs, @functor
using CUDA
using Base:@kwdef

include("./params.jl")
include("./types.jl")
include("./tools.jl")
include("./couplingLayers.jl")

export DeviceParams, ActionParams, ModelParams
export Configuration, Phi4ScalarAction
export sampleNorm, freezing_mask
export AffineCoupling, create_conv_net

end # module
