### Fit a full rank model with PRISMA
import FirstOrderOptimization: PRISMA, PrismaParams, PrismaStepsize

export PrismaParams, PrismaStepsize, fit!

export GFRM

# todo
# * check syntactic correctness
# * estimate lipshitz constant more reasonably
# * map GFRM to GLRM and back
# * implement trace norm
# * check that PRISMA code calculates the right thing via SDP

type GFRM{L<:Loss, R<:ProductRegularizer}<:AbstractGLRM
    A                            # The data table
    losses::Array{L,1}           # Array of loss functions
    r::R                         # The regularization to be applied to U
    k::Int                       # Estimated rank of solution U
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
    U::AbstractArray{Float64,2}  # Representation of data in numerical space. A ≈ U = X'Y
end

### FITTING
function fit!(gfrm::GFRM, params::PrismaParams = PrismaParams(PrismaStepsize(1), 100, 1);
			  ch::ConvergenceHistory=ConvergenceHistory("PrismaGFRM"), 
			  verbose=true,
			  kwargs...)

    # W will be the symmetric parameter; U is the upper right block
    U(W) = W[1:m, m+1:end]

    # we're closing over yidxs and gfrm and m and n
    yidxs = get_yidxs(gfrm.losses)
    m,n = size(gfrm.A)

    ## Grad of f
    function grad_f(W)
        G = zeros(m,n)
        Umat = U(W)
        for j=1:n
            for i in gfrm.observed_examples[j]
                G[i,yidxs[j]] = .5*grad(gfrm.losses[j], Umat[i,yidxs[j]], gfrm.A[i,j])
            end
        end
        return [zeros(m,m) G; G' zeros(n,n)]
    end

    ## Prox of g
    prox_g(W, alpha) = prox(gfrm.r, W, alpha)

    ## Prox of h
    # we're going to use a closure over gfrm.k
    # to remember what the rank of prox_h(W) was the last time we computed it
    # in order to avoid calculating too many eigentuples of W
    function prox_h(W, alpha=0; TOL=1e-10)
        while gfrm.k < size(W,1)
            l,v = eigs(Symmetric(W), nev = gfrm.k+1, which=:LR) # v0 = [v zeros(size(W,1), gfrm.k+1 - size(v,2))]
            if l[end] <= TOL
                gfrm.k = sum(l.>=TOL)
                return v*diagm(max(l,0))*v'
            else
                gfrm.k = 2*gfrm.k # double the rank and try again
            end
        end
        # else give up on computational cleverness
        l,v = eig(Symmetric(W))
        gfrm.k = sum(l.>=TOL)
        return v*diagm(max(l,0))*v'
    end

    ## Objective evaluation
    # we're not going to bother checking whether W is psd or not
    # when evaluating the objective; in the course of the prisma
    # algo this makes no difference
    function obj(W)
        Umat = U(W)
        err = 0.0
        for j=1:n
            for i in gfrm.observed_examples[j]
                err += evaluate(gfrm.losses[j], Umat[i,yidxs[j]], gfrm.A[i,j])
            end
        end
        err += evaluate(gfrm.r, W)
        return err
    end

    # initialize
    W = zeros(m+n,m+n)
    # lipshitz constant for f (XXX right now a wild guess that makes sense for unscaled quadratic loss)
    L_f = 2
    # orabona starts stepsize at
    # beta = lambda/sqrt((m+n)^2*mean(A.^2))
    params.stepsizerule.initial_stepsize = gfrm.r.scale/sqrt(obj(W))

    # recover
    t = time()
    W = PRISMA(W, L_f,
           grad_f,
           prox_g,
           prox_h,
           obj,
           params)

    # t = time() - t
    # update!(ch, t, obj(W)) 

    return gfrm.U, ch
end
