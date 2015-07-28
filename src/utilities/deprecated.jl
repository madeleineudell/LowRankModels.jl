using Base.depwarn

# Julia .4
if VERSION >= v"0.4"
	Base.@deprecate GLRM(A::AbstractArray, obs::Array{Tuple(Int, Int), 1}, args...; kwargs...) GLRM(A, args...; obs = obs, kwargs...)
# Julia .3
else
	#Base.@deprecate 
	GLRM(A::AbstractArray, obs::Array{(Int, Int), 1}, args...; kwargs...) = GLRM(A, args...; obs = obs, kwargs...)
end