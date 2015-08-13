export fit, fit!, Params

### PARAMETERS TYPE
abstract AbstractParams
Params(args...; kwargs...) = ProxGradParams(args...; kwargs...)

# default in-place fitting uses proximal gradient method
@compat function fit!(glrm::AbstractGLRM; kwargs...)
    kwdict = Dict(kwargs)
    if :params in keys(kwdict)
        return fit!(glrm, kwdict[:params]; kwargs...)
    else
        return fit!(glrm, ProxGradParams(); kwargs...)
    end
end

# fit without modifying the glrm object
function fit(glrm::AbstractGLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fit!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X',Y,ch
end