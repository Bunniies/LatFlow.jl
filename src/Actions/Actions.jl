module Actions

using Flux

include("./phi4.jl")
export Phi4ScalarAction

include("xy.jl")
export XYmodelAction

end