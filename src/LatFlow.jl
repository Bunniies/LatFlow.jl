module LatFlow

using Distributions, CircularArrays, PyPlot, Random
using Flux
using Flux: onehotbatch, onecold, @epochs, @functor
using Base:@kwdef

include("./params.jl")
include("./types.jl")
include("./tools.jl")

export ActionParams
export Configuration, Phi4ScalarAction
export sampleNorm, freezing_mask

end # module
