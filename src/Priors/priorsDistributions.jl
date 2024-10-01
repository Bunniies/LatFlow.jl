function get_prior(type::String="Normal"; mu=0.0f0, k=8.0f0, a=0, b=1)
   
    if type == "Normal"
        prior = Normal{Float32}(0.f0, 1.f0)
    elseif type == "NormalFiniteSupp"
        prior = Normal{Float32}(Float32(pi), 0.5f0)
    elseif type == "VonMises"
        prior = VonMises(mu, k)
    elseif type == "MultivariateUniform"
        prior = Uniform(a, b)
    else
        error("Prior distribution of type $(type) is not supported.")
    end

    return prior
end

