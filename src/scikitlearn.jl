import ScikitLearnBase
using ScikitLearnBase: @declare_hyperparameters

export SkGLRM, PCA, QPCA, NNMF, KMeans, RPCA

################################################################################
# Shared definitions

# Note: there is redundancy in the hyperparameters. This is
# necessary if we want to offer a simple interface in PCA(), and a full
# interface in SkGLRM(). PCA(abs_tol=0.1, max_iter=200) cannot create
# `ProxGradParams(abs_tol, max_iter)` right away, because abs_tol and
# max_iter are hyperparameters and need to be visible/changeable by
# set_params for grid-search.
# There are other ways of setting it up, but this seems like the simplest.
type SkGLRM <: ScikitLearnBase.BaseEstimator
    # Hyperparameters: those will be passed to GLRM, so it doesn't matter if
    # they're not typed.
    fit_params # if fit_params != nothing, it has priority over abs_tol, etc.
    loss
    rx
    ry
    # rx/ry_scale can be nothing, in which case they're ignored. This allows
    # ry to be a vector
    rx_scale
    ry_scale
    abs_tol::Float64
    rel_tol::Float64
    max_iter::Int
    inner_iter::Int
    k::Int
    init::Function   # initialization function
    verbose::Bool

    glrm::GLRM     # left undefined by the constructor
end

# This defines `clone`, `get_params` and `set_params!`
@declare_hyperparameters(SkGLRM, [:fit_params, :init, :rx, :ry,
                                 :rx_scale, :ry_scale, :loss,
                                 :abs_tol, :rel_tol, :max_iter, :inner_iter, :k,
                                 :verbose])

function do_fit!(skglrm::SkGLRM, glrm::GLRM)
    fit_params = (skglrm.fit_params === nothing ?
                  ProxGradParams(abs_tol=skglrm.abs_tol,
                                 rel_tol=skglrm.rel_tol,
                                 max_iter=skglrm.max_iter) :
                  skglrm.fit_params)

    fit!(glrm, fit_params; verbose=skglrm.verbose)
end

function build_glrm(skglrm::SkGLRM, X, missing_values)
    k = skglrm.k == -1 ? size(X, 2) : skglrm.k
    obs = [ind2sub(missing_values, x) for x in find(.!missing_values)]
    rx, ry = skglrm.rx, skglrm.ry
    if skglrm.rx_scale !== nothing
        rx = copy(rx)
        scale!(rx, skglrm.rx_scale)
    end
    if skglrm.ry_scale !== nothing
        ry = copy(ry)
        scale!(ry, skglrm.ry_scale)
    end
    GLRM(X, skglrm.loss, rx, ry, k; obs=obs)
end

# The input matrix is called X (instead of A) following ScikitLearn's convention
function ScikitLearnBase.fit_transform!(skglrm::SkGLRM, X, y=nothing;
                                        missing_values=isnan.(X))
    @assert size(X)==size(missing_values)

    # Reuse the standard GLRM constructor and fitting machinery
    skglrm.glrm = build_glrm(skglrm, X, missing_values)
    skglrm.init(skglrm.glrm)
    X, _, _ = do_fit!(skglrm, skglrm.glrm)

    return X'
end

function ScikitLearnBase.fit!(skglrm::SkGLRM, X, y=nothing; kwargs...)
    ScikitLearnBase.fit_transform!(skglrm, X; kwargs...)
    skglrm
end


""" `transform(skglrm::SkGLRM, X)` brings X to low-rank-space """
function ScikitLearnBase.transform(skglrm::SkGLRM, X;
                                   missing_values=isnan.(X))
    glrm = skglrm.glrm
    ry_fixed = [FixedLatentFeaturesConstraint(glrm.Y[:, i])
                for i=1:size(glrm.Y, 2)]
    glrm_fixed = build_glrm(skglrm, X, missing_values)
    X2, _, ch = do_fit!(skglrm, glrm_fixed)
    return X2'
end


""" `transform(skglrm::SkGLRM, X)` brings X from low-rank-space back to the
original input-space """
ScikitLearnBase.inverse_transform(skglrm::SkGLRM, X) = X * skglrm.glrm.Y

# Only makes sense for KMeans
function ScikitLearnBase.predict(km::SkGLRM, X)
    X2 = ScikitLearnBase.transform(km, X)
    # This performs the "argmax" over the columns to get the cluster #
    return mapslices(indmax, X2, 2)[:]
end

################################################################################
# Public constructors

"""
    SkGLRM(; fit_params=nothing, init=glrm->nothing, k::Int=-1,
           loss=QuadLoss(), rx::Regularizer=ZeroReg(), ry=ZeroReg(),
           rx_scale=nothing, ry_scale=nothing,
           # defaults taken from proxgrad.jl
           abs_tol=0.00001, rel_tol=0.0001, max_iter=100, inner_iter=1,
           verbose=false)

Generalized low rank model (GLRM). GLRMs model a data array by a low rank
matrix. GLRM makes it easy to mix and match loss functions and regularizers to
construct a model suitable for a particular data set.

Hyperparameters:

- `fit_params`: algorithm to use in fitting the GLRM. Defaults to
   `ProxGradParams(abs_tol, rel_tol, skglrm.max_iter)`
- `init`: function to initialize the low-rank matrices, before the main gradient
   descent loop.
- `k`: number of components (rank of the latent representation). By default,
   use k=nfeatures (full rank)
- `loss`: loss function. Can be either a single `::Loss` object, or a vector
   of `nfeature` loss objects, allowing for mixed inputs (eg. binary and
   continuous data)
- `rx`: regularization over the hidden coefficient matrix
- `ry`: regularization over the latent features matrix. Can be either a single
   regularizer, or a vector of regularizers of length nfeatures, allowing
   for mixed inputs
- `rx_scale`, `ry_scale`: strength of the regularization (higher is stronger).
   By default, `scale=1`. Cannot be used if `rx/ry` are vectors.
- `abs_tol, rel_tol`: tolerance criteria to stop the gradient descent iteration
- `max_iter, inner_iter`: number of iterations in the gradient descent loops
- `verbose`: print convergence information

All parameters (in particular, `rx/ry_scale`) can be tuned with
`ScikitLearn.GridSearch.GridSearchCV`

For more information on the parameters see [LowRankModels](https://github.com/madeleineudell/LowRankModels.jl)
"""
function SkGLRM(; fit_params=nothing, init=glrm->nothing, k=-1,
                loss=QuadLoss(), rx=ZeroReg(), ry=ZeroReg(),
                rx_scale=nothing, ry_scale=nothing,
                # defaults taken from proxgrad.jl
                abs_tol=0.00001, rel_tol=0.0001, max_iter=100, inner_iter=1,
                verbose=false)
    dummy = pca(zeros(1,1), 1) # it needs an initial value - will be overwritten
    return SkGLRM(fit_params, loss, rx, ry, rx_scale, ry_scale, abs_tol,
                  rel_tol, max_iter,
                  inner_iter, k, init, verbose, dummy)
end


"""    PCA(; k=-1, ...)

Principal Component Analysis with `k` components (defaults to using
`nfeatures`). Equivalent to

    SkGLRM(loss=QuadLoss(), rx=ZeroReg(), ry=ZeroReg(), init=init_svd!)

See ?SkGLRM for more hyperparameters. In particular, increasing `max_iter`
(default 100) may improve convergence. """
function PCA(; kwargs...)
    # principal components analysis
    # minimize ||A - XY||^2
    loss = QuadLoss()
    r = ZeroReg()
    return SkGLRM(; loss=loss, rx=r, ry=r, init=init_svd!, kwargs...)
end


"""    QPCA(k=-1, rx_scale=1, ry_scale=1; ...)

Quadratically Regularized PCA with `k` components
(default: `k = nfeatures`). Equivalent to

    SkGLRM(loss=QuadLoss(), rx=QuadReg(1.0), ry=QuadReg(1.0), init=init_svd!)

Regularization strength is set by `rx_scale` and `ry_scale`. See ?SkGLRM for
more hyperparameters.
"""
function QPCA(; kwargs...)
    # quadratically regularized principal components analysis
    # minimize ||A - XY||^2 + rx_scale*||X||^2 + ry_scale*||Y||^2
    loss = QuadLoss()
    r = QuadReg(1.0) # scale is set in build_glrm
    return SkGLRM(; loss=loss, rx=r, ry=r, init=init_svd!, kwargs...)
end


"""    NNMF(; k=-1, ...)

Non-negative matrix factorization with `k` components (default:
`k=nfeatures`). Equivalent to

    SkGLRM(loss=QuadLoss(), rx=NonNegConstraint(), ry=NonNegConstraint(), init=init_svd!)

See ?SkGLRM for more hyperparameters
"""
function NNMF(; kwargs...)
    # nonnegative matrix factorization
    # minimize_{X>=0, Y>=0} ||A - XY||^2
    loss = QuadLoss()
    r = NonNegConstraint()
    return SkGLRM(; loss=loss,rx=r,ry=r, init=init_svd!, kwargs...)
end


"""    KMeans(; k=2, inner_iter=10, max_iter=100, ...)

K-Means algorithm. Separates the data into `k` clusters. See ?SkGLRM for more
hyperparameters. In particular, increasing `inner_iter` and `max_iter` may
improve convergence.

**IMPORTANT**: This is not the most efficient way of performing K-Means, and
the iteration may not reach convergence.
"""
function KMeans(; k=2, inner_iter=10, kwargs...)
    # minimize_{columns of X are unit vectors} ||A - XY||^2
    loss = QuadLoss()
    rx = UnitOneSparseConstraint()
    ry = ZeroReg()
    return SkGLRM(k=k, loss=loss,rx=rx,ry=ry, inner_iter=inner_iter,
                  init=init_kmeanspp!; kwargs...)
end


"""    RPCA(; k=-1, ...)

Robust PCA with `k` components (default: `k = nfeatures`). Equivalent to

    SkGLRM(loss=HuberLoss(), rx=QuadReg(1.0), ry=QuadReg(1.0), init=init_svd!)

Regularization strength is set by `rx_scale` and `ry_scale`. See ?SkGLRM for
more hyperparameters. In particular, increasing `max_iter` (default 100) may
improve convergence. """
function RPCA(; kwargs...)
    # robust PCA
    # minimize HuberLoss(A - XY) + scale*||X||^2 + scale*||Y||^2
    loss = HuberLoss()
    r = QuadReg(1.0)
    return SkGLRM(; loss=loss,rx=r,ry=r, init=init_svd!, kwargs...)
end
