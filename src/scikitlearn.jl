import ScikitLearnBase

type SkGLRM<:AbstractGLRM
    fit_params
    verbose
    init::Function   # initialization function
    
    # These fields are copied straight from GLRM's definition. They are kept to
    # achieve 100% compatibility.  The types are gone because it's a headache
    # to initialize them, and the computations are done through temporary
    # ::GLRM objects anyway.
    A                            # The data table
    losses                       # array of loss functions
    rx                           # The regularization to be applied to each row of Xᵀ (column of X)
    ry                           # Array of regularizers to be applied to each column of Y
    k::Int                       # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y

    SkGLRM(; fit_params=ProxGradParams(), init=glrm->glrm,
           losses=QuadLoss(),rx=ZeroReg(), ry=ZeroReg(), k=-1, verbose=false) = 
        new(fit_params, verbose, init, nothing, losses, rx, ry, k)
end

function PCA(; k::Int=-1, verbose=false)
    loss = QuadLoss()
    r = ZeroReg()
    # I had to pick these convergence_tol and max_iter to get reasonably low
    # reconstruction error on the Iris dataset
    fit_params = ProxGradParams(convergence_tol=1.e-8, max_iter=1000)
    return SkGLRM(fit_params=fit_params,
                  losses=loss, rx=r, ry=r, k=k, verbose=verbose,
                  init=init_svd!)
end

# The input matrix is called X (instead of A) following ScikitLearn's convention
function ScikitLearnBase.fit!(skglrm::SkGLRM, X, y=nothing)
    if skglrm.k == -1
        # Is that a good default in general? scikitlearn.PCA uses
        # min(nrows, ncols)
        skglrm.k = size(X, 2)
    end
    
    # Reuse the standard GLRM constructor and machinery
    temp = GLRM(X, skglrm.losses, skglrm.rx, skglrm.ry, skglrm.k)
    skglrm.init(temp)
    fit!(temp, skglrm.fit_params; verbose=skglrm.verbose)
    # Copy the results back to skglrm
    for field in [:A, :losses, :rx, :ry, :k, :observed_features,
                  :observed_examples, :X, :Y]
        setfield!(skglrm, field, getfield(temp, field))
    end
    
    skglrm
end

"""    `transform(skglrm::SkGLRM, X)` brings X to low-rank-space """
function ScikitLearnBase.transform(skglrm::SkGLRM, X)
    ry_fixed = [FixedLatentFeaturesConstraint(skglrm.Y[:, i])
                for i=1:size(skglrm.Y, 2)]
    # Should we copy losses, rx, etc.? In what way are they mutated?
    glrm_fixed = GLRM(X, skglrm.losses, skglrm.rx, ry_fixed, skglrm.k);
    X2, _, ch = fit!(glrm_fixed, skglrm.fit_params; verbose=skglrm.verbose);
    return X2'
end

"""    `transform(skglrm::SkGLRM, X)` brings X from low-rank-space back to the
original input-space """
ScikitLearnBase.inverse_transform(skglrm::SkGLRM, X) = X * skglrm.Y
