using Base.depwarn

@compat Base.@deprecate GLRM(A::AbstractArray, obs::Array{Tuple{Int, Int}, 1}, args...; kwargs...) GLRM(A, args...; obs = obs, kwargs...)

Base.@deprecate ProxGradParams(s::Number,m::Int,c::Float64,ms::Float64) ProxGradParams(s, max_iter=m, convergence_tol=c, min_stepsize=ms)

Base.@deprecate expand_categoricals expand_categoricals!

Base.@deprecate errors(g::GLRM) error_metric(g)