export fit, fit!, Params

### PARAMETERS TYPE
abstract type AbstractParams end
Params(args...; kwargs...) = ProxGradParams(args...; kwargs...)

# default in-place fitting uses proximal gradient method
function fit!(glrm::AbstractGLRM; kwargs...)
    kwdict = Dict(kwargs)
    if :params in keys(kwdict)
        return fit!(glrm, kwdict[:params]; kwargs...)
    else
        if isa(glrm.A,SparseMatrixCSC)
            # Default to sparse algorithm for a sparse dataset
            return fit!(glrm, SparseProxGradParams(); kwargs...)
        else
            # Classic proximal gradient method for non-sparse data
            return fit!(glrm, ProxGradParams(); kwargs...)
        end
    end
end

# fit without modifying the glrm object
function fit(glrm::AbstractGLRM, args...; kwargs...)
    X0 = Array{Float64}(undef, size(glrm.X))
    Y0 = Array{Float64}(undef, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fit!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X',Y,ch
end
