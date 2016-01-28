### Fit a full rank model with PRISMA
import FirstOrderMethods: prisma, PrismaParams, PrismaStepsize

export PrismaParams, PrismaStepsize, fit!

export GFRM

# todo
# * check syntactic correctness
# * estimate lipshitz constant more reasonably
# * map GFRM to GLRM and back
# * implement trace norm
# * check that PRISMA code calculates the right thing via SDP

type GFRM{L<:Loss, R<:Regularizer}<:AbstractGLRM
    A                            # The data table
    losses::Array{L,1}           # array of loss functions
    r::Regularizer               # The regularization to be applied to U
    k::Int                       # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
    U::AbstractArray{Float64,2}  # Representation of data in numerical space. A ≈ U = X'Y
end

### FITTING
function fit!(glrm::GLRM, params::PrismaParams = PrismaParams(PrismaStepsize(1), 100, 1);
			  ch::ConvergenceHistory=ConvergenceHistory("ProxGradGLRM"), 
			  verbose=true,
			  kwargs...)

    # W will be the symmetric parameter; U is the upper right block
    U(W) = W[1:m, m+1:end]

    # we're closing over yidxs and glrm and m and n
    yidxs = get_yidxs(glrm.losses)
    m,n = size(glrm.A)

    ## Grad of f
    function grad_f(W)
        G = zeros(size(W))
        for j=1:n
            for i in glrm.observed_examples[j]
                G[i,yidxs[j]] = grad(glrm.losses[j], W[i,yidxs[j]], glrm.A[i,j])
            end
        end
        return [zeros(m,m) G; G' zeros(n,n)]
    end

    ## Prox of g
    prox_g(W, alpha) = prox(glrm.r, W, alpha)

    ## Prox of h
    # we're going to use a closure over prevrank
    # to remember what the rank of prox_h(W) was the last time we computed it
    # in order to avoid calculating too many eigentuples of W
    prevrank = PrevRank(k)
    function prox_h(W, alpha=0; TOL=1e-10)
        while prevrank.r < size(W,1)
            l,v = eigs(Symmetric(W), nev = prevrank.r+1, which=:LR) # v0 = [v zeros(size(W,1), prevrank.r+1 - size(v,2))]
            if l[end] <= TOL
                prevrank.r = sum(l.>=TOL)
                return v*diagm(max(l,0))*v'
            else
                prevrank.r = 2*prevrank.r # double the rank and try again
            end
        end
        # else give up on computational cleverness
        l,v = eig(Symmetric(W))
        prevrank.r = sum(l.>=TOL)
        return v*diagm(max(l,0))*v'
    end

    ## Objective evaluation
    # we're not going to bother checking whether W is psd or not
    # when evaluating the objective; in the course of the prisma
    # algo this makes no difference
    function obj(W)
        err = 0.0
        for j=1:n
            for i in glrm.observed_examples[j]
                err += evaluate(glrm.losses[j], W[i,yidxs[j]], glrm.A[i,j])
            end
        end
        err += evaluate(glrm.r, W)
        return err
    end

    # initialize
    W = zeros(m+n,m+n)
    # lipshitz constant for f (right now a wild guess)
    L_f = 2
    # orabona starts stepsize at
    # beta = lambda/sqrt((m+n)^2*mean(A.^2))
    params.stepsizerule.initial_stepsize = glrm.r.scale/sqrt(obj(W))

    # recover
    W = PRISMA(W, L_f,
           grad_f,
           prox_g,
           prox_h,
           obj,
           params)

    t = time() - t
    update!(ch, t, obj(W))

    # return X and Y
    while prevrank.r < size(W,1)
        l,v = eigs(Symmetric(W), nev = prevrank.r+1, which=:LR) # v0 = [v zeros(size(W,1), prevrank.r+1 - size(v,2))]
        if l[end] <= TOL
            prevrank.r = sum(l.>=TOL)
            return v*diagm(max(l,0))*v'
        else
            prevrank.r = 2*prevrank.r # double the rank and try again
        end
    end    

    return glrm.X, glrm.Y, ch
end
