import Base: size, axpy!
import Base.LinAlg.scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView

export GLRM, getindex, size,
       objective, error_metric, 
       add_offset!, equilibrate_variance!, fix_latent_features!,
       impute, error_metric

ObsArray = Union(Array{Array{Int,1},1}, Array{UnitRange{Int},1})

### GLRM TYPE
type GLRM{L<:Loss, R<:Regularizer}
    A::AbstractArray             # The data table transformed into a coded array 
    losses::Array{L,1}           # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{R,1}               # Array of regularizers to be applied to each column of Y
    k::Int                       # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::AbstractArray{Float64,2}          # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}          # Representation of features in low-rank space. A ≈ X'Y
end

# usage notes:
# * providing argument `obs` overwrites arguments `observed_features` and `observed_examples`
# * offset and scale are *false* by default to avoid unexpected behavior
# * convenience methods for calling are defined in utilities/conveniencemethods.jl
function GLRM(A::AbstractArray, losses::Array, rx::Regularizer, ry::Array, k::Int; 
# the following tighter definition fails when you form an array of a tighter subtype than the abstract type, eg Array{quadratic,1}
# function GLRM(A::AbstractArray, losses::Array{Loss,1}, rx::Regularizer, ry::Array{Regularizer,1}, k::Int; 
              X = randn(k,size(A,1)), Y = randn(k,size(A,2)),
              obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
              observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
              observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
              offset = false, scale = false,
              checknan = true)
    # Check dimensions of the arguments
    m,n = size(A)
    if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
    if length(ry)!=n error("There must be either one Y regularizer or as many Y regularizers as there are columns in the data matrix") end
    if size(X)!=(k,m) error("X must be of size (k,m) where m is the number of rows in the data matrix. This is the transpose of the standard notation used in the paper, but it makes for better memory management. size(X) = $(size(X)), size(A) = $(size(A))") end
    if size(Y)!=(k,n) error("Y must be of size (k,n) where n is the number of columns in the data matrix. size(Y) = $(size(Y)), size(A) = $(size(A))") end
    if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
        # println("no obs given, using observed_features and observed_examples")
        glrm = GLRM(A,losses,rx,ry,k, observed_features, observed_examples, X,Y)
    else # otherwise unpack the tuple list into arrays
        # println("unpacking obs into array")
        glrm = GLRM(A,losses,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y)
    end

    # check to make sure X is properly oriented
    if size(glrm.X) != (k, size(A,1)) 
        # println("transposing X")
        glrm.X = glrm.X'
    end
    # check none of the observations are NaN
    if checknan
        for i=1:size(A,1)
            for j=glrm.observed_features[i]
                if isnan(A[i,j]) 
                    error("Observed value in entry ($i, $j) is NaN.")
                end
            end
        end
    end

    if scale # scale losses (and regularizers) so they all have equal variance
        equilibrate_variance!(glrm)
    end
    if offset # don't penalize the offset of the columns
        add_offset!(glrm)
    end
    return glrm
end

### OBSERVATION TUPLES TO ARRAYS
@compat function sort_observations(obs::Array{Tuple{Int,Int},1}, m::Int, n::Int; check_empty=false)
    observed_features = Array{Int,1}[Int[] for i=1:m]
    observed_examples = Array{Int,1}[Int[] for j=1:n]
    for (i,j) in obs
        @inbounds push!(observed_features[i],j)
        @inbounds push!(observed_examples[j],i)
    end
    if check_empty && (any(map(x->length(x)==0,observed_examples)) || 
            any(map(x->length(x)==0,observed_features)))
        error("Every row and column must contain at least one observation")
    end
    return observed_features, observed_examples
end

## SCALINGS AND OFFSETS ON GLRM
function add_offset!(glrm::GLRM)
    glrm.rx, glrm.ry = lastentry1(glrm.rx), map(lastentry_unpenalized, glrm.ry)
    return glrm
end
function equilibrate_variance!(glrm::GLRM)
    for i=1:size(glrm.A,2)
        nomissing = glrm.A[glrm.observed_examples[i],i]
        if length(nomissing)>0
            varlossi = avgerror(glrm.losses[i], nomissing)
            varregi = var(nomissing) # TODO make this depend on the kind of regularization; this assumes quadratic
        else
            varlossi = 1
            varregi = 1
        end
        if varlossi > 0
            # rescale the losses and regularizers for each column by the inverse of the empirical variance
            scale!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
        end
        if varregi > 0
            scale!(glrm.ry[i], scale(glrm.ry[i])/varregi)
        end
    end
    return glrm
end
function fix_latent_features!(glrm::GLRM, n)
    glrm.ry = Regularizer[fixed_latent_features(glrm.ry[i], glrm.Y[1:n,i]) 
                            for i in 1:length(glrm.ry)]
    return glrm
end

## ERROR METRIC EVALUATION (BASED ON DOMAINS OF THE DATA)
function raw_error_metric(glrm::GLRM, XY::Array{Float64,2}, domains::Array{Domain,1})
    m,n = size(glrm.A)
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += error_metric(domains[j], glrm.losses[j], XY[i,j], glrm.A[i,j])
        end
    end
    return err
end
function std_error_metric(glrm::GLRM, XY::Array{Float64,2}, domains::Array{Domain,1})
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
function error_metric(glrm::GLRM, XY::Array{Float64,2}, domains::Array{Domain,1}; standardize=false)
    if standardize
        return std_error_metric(glrm, XY, domains)
    else
        return raw_error_metric(glrm, XY, domains)
    end
end
# The user can also pass in X and Y and `error_metric` will compute XY for them
function error_metric(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}, domains::Array{Domain,1}; kwargs...)
    XY = Array(Float64, size(glrm.A)) 
    gemm!('T','N',1.0,X,Y,0.0,XY) 
    error_metric(glrm, XY, domains; kwargs...)
end
# Or just the GLRM and `error_metric` will use glrm.X and .Y
error_metric(glrm::GLRM, domains::Array{Domain,1}; kwargs...) = error_metric(glrm, glrm.X, glrm.Y, domains; kwargs...)
error_metric(glrm::GLRM; kwargs...) = error_metric(glrm, Domain[l.domain for l in glrm.losses]; kwargs...)

# Use impute and errors over GLRMS
impute(glrm::GLRM) = impute(glrm.losses, glrm.X'*glrm.Y)