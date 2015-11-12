export objective, error_metric, impute

### OBJECTIVE FUNCTION EVALUATION FOR MPCA
function objective(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2}, 
                   XY::Array{Float64,2}; 
                   yidxs = 1:size(A,2), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += evaluate(glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        err += calc_penalty(glrm,X,Y)
    end
    return err
end
# The user can also pass in X and Y and `objective` will compute XY for them
function objective(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2};
                   sparse=false, include_regularization=true)
    XY = Array(Float64, size(glrm.A)) 
    if sparse
        # Calculate X'*Y only at observed entries of A
        m,n = size(glrm.A)
        err = 0.0
        for j=1:n
            for i in glrm.observed_examples[j]
                err += evaluate(glrm.losses[j], dot(X[:,i],Y[:,j]), glrm.A[i,j])
            end
        end
        if include_regularization
            err += calc_penalty(glrm,X,Y)
        end
        return err
    else
        # dense calculation variant (calculate XY up front)
        gemm!('T','N',1.0,X,Y,0.0,XY)
        return objective(glrm, X, Y, XY; include_regularization=include_regularization)
    end
end
# Or just the GLRM and `objective` will use glrm.X and .Y
objective(glrm::AbstractGLRM; kwargs...) = objective(glrm, glrm.X, glrm.Y; kwargs...)

# For shared arrays
# TODO: compute objective in parallel
objective(glrm::ShareGLRM, X::SharedArray{Float64,2}, Y::SharedArray{Float64,2}) =
    objective(glrm, X.s, Y.s)

# Helper function to calculate the regularization penalty for X and Y
function calc_penalty(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2})
    m,n = size(glrm.A)
    penalty = 0.0
    for i=1:m
        penalty += evaluate(glrm.rx, view(X,:,i))
    end
    for j=1:n
        penalty += evaluate(glrm.ry[j], view(Y,:,j))
    end
    return penalty
end

## ERROR METRIC EVALUATION (BASED ON DOMAINS OF THE DATA)
function raw_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1})
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += error_metric(domains[j], glrm.losses[j], XY[i,j], glrm.A[i,j])
        end
    end
    return err
end
function std_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1})
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        column_mean = 0.0
        column_err = 0.0
        for i in glrm.observed_examples[j]
            column_mean += glrm.A[i,j]^2
            column_err += error_metric(domains[j], glrm.losses[j], XY[i,j], glrm.A[i,j])
        end
        column_mean = column_mean/length(glrm.observed_examples[j])
        if column_mean != 0
            column_err = column_err/column_mean
        end
        err += column_err
    end
    return err
end
function error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1}; standardize=false)
    if standardize
        return std_error_metric(glrm, XY, domains)
    else
        return raw_error_metric(glrm, XY, domains)
    end
end
# The user can also pass in X and Y and `error_metric` will compute XY for them
function error_metric(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2}, domains::Array{Domain,1}=Domain[l.domain for l in glrm.losses]; kwargs...)
    XY = Array(Float64, size(glrm.A)) 
    gemm!('T','N',1.0,X,Y,0.0,XY) 
    error_metric(glrm, XY, domains; kwargs...)
end
# Or just the GLRM and `error_metric` will use glrm.X and .Y
error_metric(glrm::AbstractGLRM, domains::Array{Domain,1}; kwargs...) = error_metric(glrm, glrm.X, glrm.Y, domains; kwargs...)
error_metric(glrm::AbstractGLRM; kwargs...) = error_metric(glrm, Domain[l.domain for l in glrm.losses]; kwargs...)

# Use impute and errors over GLRMS
impute(glrm::AbstractGLRM) = impute(glrm.losses, glrm.X'*glrm.Y)