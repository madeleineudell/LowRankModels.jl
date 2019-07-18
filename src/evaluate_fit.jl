export objective, error_metric, impute, impute_missing

### OBJECTIVE FUNCTION EVALUATION FOR MPCA
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2},
                   XY::Array{Float64,2};
                   yidxs = get_yidxs(glrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += evaluate(glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        err += calc_penalty(glrm,X,Y; yidxs = yidxs)
    end
    return err
end
function row_objective(glrm::AbstractGLRM, i::Int, x::AbstractArray, Y::Array{Float64,2} = glrm.Y;
                   yidxs = get_yidxs(glrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    err = 0.0
    XY = x'*Y
    for j in glrm.observed_features[i]
        err += evaluate(glrm.losses[j], XY[1,yidxs[j]], glrm.A[i,j])
    end
    # add regularization penalty
    if include_regularization
        err += evaluate(glrm.rx[i], x)
    end
    return err
end
function col_objective(glrm::AbstractGLRM, j::Int, y::AbstractArray, X::Array{Float64,2} = glrm.X;
                   include_regularization=true)
    m,n = size(glrm.A)
    sz = size(y)
    if length(sz) == 1 colind = 1 else colind = 1:sz[2] end
    err = 0.0
    XY = X'*y
    obsex = glrm.observed_examples[j]
    @inbounds XYj = XY[obsex,colind]
    @inbounds Aj = convert(Array, glrm.A[obsex,j])
    err += evaluate(glrm.losses[j], XYj, Aj)
    # add regularization penalty
    if include_regularization
        err += evaluate(glrm.ry[j], y)
    end
    return err
end
# The user can also pass in X and Y and `objective` will compute XY for them
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2};
                   sparse=false, include_regularization=true,
                   yidxs = get_yidxs(glrm.losses), kwargs...)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,size(glrm.A,1)))
    XY = Array{Float64}(undef, (size(X,2), size(Y,2)))
    if sparse
        # Calculate X'*Y only at observed entries of A
        m,n = size(glrm.A)
        err = 0.0
        for j=1:n
            for i in glrm.observed_examples[j]
                err += evaluate(glrm.losses[j], dot(X[:,i],Y[:,yidxs[j]]), glrm.A[i,j])
            end
        end
        if include_regularization
            err += calc_penalty(glrm,X,Y; yidxs = yidxs)
        end
        return err
    else
        # dense calculation variant (calculate XY up front)
        gemm!('T','N',1.0,X,Y,0.0,XY)
        return objective(glrm, X, Y, XY; include_regularization=include_regularization, yidxs = yidxs, kwargs...)
    end
end
# Or just the GLRM and `objective` will use glrm.X and .Y
objective(glrm::GLRM; kwargs...) = objective(glrm, glrm.X, glrm.Y; kwargs...)

# For shared arrays
# TODO: compute objective in parallel
objective(glrm::ShareGLRM, X::SharedArray{Float64,2}, Y::SharedArray{Float64,2}) =
    objective(glrm, X.s, Y.s)

# Helper function to calculate the regularization penalty for X and Y
function calc_penalty(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    penalty = 0.0
    for i=1:m
        penalty += evaluate(glrm.rx[i], view(X,:,i))
    end
    for f=1:n
        penalty += evaluate(glrm.ry[f], view(Y,:,yidxs[f]))
    end
    return penalty
end

## ERROR METRIC EVALUATION (BASED ON DOMAINS OF THE DATA)
function raw_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += error_metric(domains[j], glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    return err
end
function std_error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        column_mean = 0.0
        column_err = 0.0
        for i in glrm.observed_examples[j]
            column_mean += glrm.A[i,j]^2
            column_err += error_metric(domains[j], glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
        column_mean = column_mean/length(glrm.observed_examples[j])
        if column_mean != 0
            column_err = column_err/column_mean
        end
        err += column_err
    end
    return err
end
function error_metric(glrm::AbstractGLRM, XY::Array{Float64,2}, domains::Array{Domain,1};
    standardize=false,
    yidxs = get_yidxs(glrm.losses))
    m,n = size(glrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    if standardize
        return std_error_metric(glrm, XY, domains; yidxs = yidxs)
    else
        return raw_error_metric(glrm, XY, domains; yidxs = yidxs)
    end
end
# The user can also pass in X and Y and `error_metric` will compute XY for them
function error_metric(glrm::AbstractGLRM, X::Array{Float64,2}, Y::Array{Float64,2}, domains::Array{Domain,1}=Domain[l.domain for l in glrm.losses]; kwargs...)
    XY = Array{Float64}((size(X,2), size(Y,2)))
    gemm!('T','N',1.0,X,Y,0.0,XY)
    error_metric(glrm, XY, domains; kwargs...)
end
# Or just the GLRM and `error_metric` will use glrm.X and .Y
error_metric(glrm::AbstractGLRM, domains::Array{Domain,1}; kwargs...) = error_metric(glrm, glrm.X, glrm.Y, domains; kwargs...)
error_metric(glrm::AbstractGLRM; kwargs...) = error_metric(glrm, Domain[l.domain for l in glrm.losses]; kwargs...)

# Use impute and errors over GLRMS
impute(glrm::AbstractGLRM) = impute(glrm.losses, glrm.X'*glrm.Y)
function impute_missing(glrm::AbstractGLRM)
  Ahat = impute(glrm)
  for j in 1:size(glrm.A,2)
    for i in glrm.observed_examples[j]
      Ahat[i,j] = glrm.A[i,j]
    end
  end
  return Ahat
end
