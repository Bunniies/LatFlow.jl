using LatFlow, BSON, Flux, CUDA

path_models = "/Users/alessandroconigli/.julia/dev/LatFlow/examples/trained_models/phi4"

ens = ["12x12"]

models = Vector(undef, length(ens))

for k in eachindex(ens)
    fname = filter( x-> occursin("best_acc.bson", x) , readdir(joinpath(path_models, ens[k]), join=true))[1]
    println(fname)
    BSON.@load fname affine_layers
    affine_cpu = affine_layers #|> Flux.cpu
    models[k] =  affine_cpu
end