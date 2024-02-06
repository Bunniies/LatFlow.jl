using Revise
using LatFlow
using TimerOutputs, Distributions, Random


batch_size=64
lat_shape=(8,8)
prior = Normal{Float32}(0.f0, 1.f0)
hp = HyperParams()

nfws = NFworkspace{8,64}(Float32,hp)
nfws.xpr4d

layer = create_affine_layers(hp, nfws=nfws)


for k in 1:200
    evolve_prior_with_flow(prior, layer, device=hp.dp.device, nfws=nfws)
end
##


print_timer(linechars=:ascii)

##






















##
batch_size=64
lat_shape=(8,8)
prior = Normal{Float32}(0.f0, 1.f0)

function fill_arr!(arr)
    rand!(prior, arr)
end

for k in 1:1e5
    @timeit "With ws" begin
        #fill_arr!(ws.xpr)
        #rand!(prior, nfws.xpr3d )
        nfws.xpr3d .= Flux.unsqueeze(rand())
        # nfws.xpr3d .= 5.
        # ws.xpr[:,:,:] = rand(prior, lat_shape..., batch_size)
    end
end

for k in 1:1e5
    @timeit "No ws" begin
        x_pr =  rand(prior, lat_shape...,1,  batch_size)
    end
end

print_timer(linechars=:ascii)
