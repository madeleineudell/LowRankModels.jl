import Base: size, axpy!
import Base.LinAlg: scale!
import Base.BLAS: gemm!
import Base: shmem_rand, shmem_randn

export ShareGLRM, share

### GLRM TYPE
type ShareGLRM{L<:Loss, R<:Regularizer}<:AbstractGLRM
    A::SharedArray               # The data table transformed into a coded array
    losses::Array{L,1}           # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{R,1}               # Array of regularizers to be applied to each column of Y
    k::Int                       # Desired rank
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
    X::SharedArray{Float64,2}    # Representation of data in low-rank space. A ≈ X'Y
    Y::SharedArray{Float64,2}    # Representation of features in low-rank space. A ≈ X'Y
end

function share(glrm::GLRM)
    isa(glrm.A, SharedArray) ? A = glrm.A : A = convert(SharedArray,glrm.A)
    isa(glrm.X, SharedArray) ? X = glrm.X : X = convert(SharedArray, glrm.X)
    isa(glrm.Y, SharedArray) ? Y = glrm.Y : Y = convert(SharedArray, glrm.Y)
    return ShareGLRM(A, glrm.losses, glrm.rx, glrm.ry, glrm.k,
                     glrm.observed_features, glrm.observed_examples,
                     X, Y)
end

### todo: define objective for shared arrays so it's evaluated (safely) in parallel
