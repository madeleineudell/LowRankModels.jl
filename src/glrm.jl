import Base: size, axpy!
import Base.LinAlg.scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView

abstract AbstractGLRM

export AbstractGLRM, GLRM, getindex, size

typealias ObsArray Union(Array{Array{Int,1},1}, Array{UnitRange{Int},1})

### GLRM TYPE
type GLRM{L<:Loss, R<:Regularizer}<:AbstractGLRM
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