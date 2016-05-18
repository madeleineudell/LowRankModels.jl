import Base: size, axpy!
import Base.LinAlg.scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView

abstract AbstractGLRM

export AbstractGLRM, GLRM, getindex, size, scale_regularizer!

typealias ObsArray @compat(Union{Array{Array{Int,1},1}, Array{UnitRange{Int},1}})

### GLRM TYPE
type GLRM<:AbstractGLRM
    A                            # The data table
    losses::Array{Loss,1}        # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{Regularizer,1}     # Array of regularizers to be applied to each column of Y
    k::Int                       # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
end

# Initialize with nothing; could be useful for copying
# GLRM{L<:Loss, R<:Regularizer}() = GLRM([],Loss[],ZeroReg(),Regularizer[],0,UnitRange{Int}[],UnitRange{Int}[],Array(Float64,(0,0)),Array(Float64,(0,0)))

# usage notes:
# * providing argument `obs` overwrites arguments `observed_features` and `observed_examples`
# * offset and scale are *false* by default to avoid unexpected behavior
# * convenience methods for calling are defined in utilities/conveniencemethods.jl
function GLRM(A, losses::Array, rx::Regularizer, ry::Array, k::Int; 
# the following tighter definition fails when you form an array of a tighter subtype than the abstract type, eg Array{QuadLoss,1}
# function GLRM(A::AbstractArray, losses::Array{Loss,1}, rx::Regularizer, ry::Array{Regularizer,1}, k::Int; 
              X = randn(k,size(A,1)), Y = randn(k,embedding_dim(losses)),
              obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
              observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
              observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
              offset = false, scale = false,
              checknan = true, sparse_na = true)
    # Check dimensions of the arguments
    m,n = size(A)
    if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
    if length(ry)!=n error("There must be either one Y regularizer or as many Y regularizers as there are columns in the data matrix") end
    if size(X)!=(k,m) error("X must be of size (k,m) where m is the number of rows in the data matrix. This is the transpose of the standard notation used in the paper, but it makes for better memory management. \nsize(X) = $(size(X)), size(A) = $(size(A)), k = $k") end
    if size(Y)!=(k,sum(map(embedding_dim, losses))) error("Y must be of size (k,d) where d is the sum of the embedding dimensions of all the losses. \n(1 for real-valued losses, and the number of categories for categorical losses).") end
    
    # Determine observed entries of data
    if obs==nothing && sparse_na && isa(A,SparseMatrixCSC)
        I,J = findn(A) # observed indices (vectors)
        obs = [(I[a],J[a]) for a = 1:length(I)] # observed indices (list of tuples)
    end
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

parameter_estimate(glrm::GLRM) = (glrm.X, glrm.Y)


function scale_regularizer!(glrm::GLRM, newscale::Number)
    scale!(glrm.rx, newscale)
    scale!(glrm.ry, newscale)
    return glrm  
end
