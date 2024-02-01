using TimerOutputs, Distributions, Random

prior = Normal{Float32}(0.f0, 1.f0)

struct Myworkspace
    xpr
    function Myworkspace(::Type{T}, lat_shape, batch_size) where {T<:AbstractFloat}
        _xpr = Array{T,3}(undef, lat_shape..., batch_size)
        return new(_xpr)
    end
end

lat_shape = (8,8)
batch_size = 64

ws = Myworkspace(Float32, lat_shape, batch_size)


function fill_arr!(arr)
    rand!(prior, arr)
    #for k in eachindex(arr)
    #    arr[k] = rand!(prior)
    #end
end

for k in 1:1e5
    @timeit "With ws" begin
        #fill_arr!(ws.xpr)
        rand!(prior, ws.xpr )
        # ws.xpr[:,:,:] = rand(prior, lat_shape..., batch_size)
    end
end

for k in 1:1e5
    @timeit "No ws" begin
        x_pr =  rand(prior, lat_shape..., batch_size)
    end
end

print_timer(linechars=:ascii)
