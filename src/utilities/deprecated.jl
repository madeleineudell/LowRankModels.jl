using Base.depwarn

@compat Base.@deprecate GLRM(A::AbstractArray, obs::Array{Tuple{Int, Int}, 1}, args...; kwargs...) GLRM(A, args...; obs = obs, kwargs...)