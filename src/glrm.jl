import Base: size, axpy!
import Base.LinAlg.scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView

export GLRM, getindex, size,
       add_offset!, equilibrate_variance!, fix_latent_features!

ObsArray = Union(Array{Array{Int,1},1}, Array{UnitRange{Int},1})

### GLRM TYPE
type GLRM
    A::AbstractArray
    losses::Array{Loss,1}        # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{Regularizer,1}     # Array of regularizers to be applied to each column of Y
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
              offset = false, scale = false)
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