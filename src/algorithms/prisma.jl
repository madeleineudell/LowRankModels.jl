### Fit a full rank model with PRISMA
import FirstOrderOptimization: PRISMA, PrismaParams, PrismaStepsize

export PrismaParams, PrismaStepsize, fit!, PrismaParams

defaultPrismaParams = PrismaParams(stepsize=PrismaStepsize(Inf), 
                                   maxiter=100, 
                                   verbose=1,
                                   reltol=1e-5)

### FITTING
function fit!(gfrm::GFRM, params::PrismaParams = defaultPrismaParams;
			  ch::ConvergenceHistory=ConvergenceHistory("PrismaGFRM"), 
			  verbose=true,
			  kwargs...)

    # we're closing over yidxs and gfrm and m and n
    yidxs = get_yidxs(gfrm.losses)
    d = maximum(yidxs[end])
    m,n = size(gfrm.A)

    # W will be the symmetric parameter; U is the upper right block
    U(W) = W[1:m, m+1:end]

    ## Grad of f
    function grad_f(W)
        G = zeros(m,d)
        Umat = U(W)
        for j=1:n
            for i in gfrm.observed_examples[j]
                # there's a 1/2 b/c 1/2 is coming from the upper right block and 1/2 from the lower left block
                G[i,yidxs[j]] = .5*grad(gfrm.losses[j], Umat[i,yidxs[j]], gfrm.A[i,j])
            end
        end
        return [zeros(m,m) G; G' zeros(d,d)]
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

    obj(W) = objective(gfrm, W, yidxs=yidxs)

    # initialize
    # lipshitz constant for f (XXX right now a wild guess that makes sense for unscaled quadratic loss)
    Lf = 2
    if params.stepsizerule.initial_stepsize == Inf
        R = sqrt(obj(gfrm.W)*m*n/sum(map(length, gfrm.observed_examples)))/max(gfrm.r.scale,1) # estimate of distance to solution
        params.stepsizerule.initial_stepsize = 2*R/Lf
    end

    # recover
    t = time()
    gfrm.W = PRISMA(gfrm.W, Lf,
           grad_f,
           prox_g,
           prox_h,
           obj,
           params)

    # t = time() - t
    # update!(ch, t, obj(W))

    gfrm.U = U(gfrm.W) 

    return gfrm.U, ch
end

## Objective evaluation
# we're not going to bother checking whether W is psd or not
# when evaluating the objective; in the course of the prisma
# algo this makes no difference
function objective(gfrm::GFRM, W::Array{Float64,2}; yidxs=get_yidxs(gfrm.losses))
    # W is the symmetric parameter; U is the upper right block
    m,n = size(gfrm.A)
    UW = W[1:m, m+1:end]
    err = 0.0
    for j=1:n
        for i in gfrm.observed_examples[j]
            err += evaluate(gfrm.losses[j], UW[i,yidxs[j]], gfrm.A[i,j])
        end
    end
    err += evaluate(gfrm.r, W)
    return err
end
function objective(gfrm::GFRM)
    objective(gfrm::GFRM, gfrm.W)
end
