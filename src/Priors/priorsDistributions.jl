function get_prior(type::String="Normal")
    if type == "Normal"
        prior = Normal{Float32}(0.f0, 1.f0)
    elseif type == "VonMisses"
        error("Prior for VonMisses not yet implemented.")
    else
        error("Prior distribution of type $(type) is not supported.")
    end
    return prior
end

