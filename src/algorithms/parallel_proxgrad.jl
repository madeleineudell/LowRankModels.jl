import Base: size, axpy!
import Base.LinAlg: scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView
import Base: shmem_rand, shmem_randn#, acontrank
export fit!, localcols

# functions for shared arrays
function localcols(Y::SharedArray)
    idxs=localindexes(Y)
    s,t=localindexes(Y)[1],localindexes(Y)[end]
    m,n=size(Y)
    return round(floor((s-1)/m+1)):round(floor(t/m))
end

### Fitting

function fit!(glrm::ShareGLRM, params::ProxGradParams;
    ch::ConvergenceHistory=ConvergenceHistory("ProxGradShareGLRM"),verbose=true)
	
	### initialization (mostly name shortening)
    isa(glrm.A, SharedArray) ? A = glrm.A : A = convert(SharedArray,glrm.A)
    # make sure that we've oriented the factors correctly
    # k, m == size(X) and k, n == size(Y)
    if size(glrm.Y,1)!==size(glrm.X,1)
        glrm.X = glrm.X'
    end
    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(glrm.Y) == 0 
        glrm.Y = .1*shmem_randn(size(glrm.Y)...) 
    end
    # at any time, glrm.X and glrm.Y will be the best model yet found, while
    # X and Y will be the working variables. All of these are shared arrays.
    X, glrm.X = glrm.X, copy(glrm.X)
    Y, glrm.Y = glrm.Y, copy(glrm.Y)

    ## a few scalars that need to be shared among all processes
    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = Base.shmem_fill(float(params.stepsize),(1,1))
    obj = Base.shmem_fill(0.0,(nprocs(),1))

    if verbose println("sending data and caching views") end
    @sync begin
        for i in procs(A)
            remotecall(i, x->(global const A=x; nothing), A)
            remotecall(i, x->(global const X=x; nothing), X)
            remotecall(i, x->(global const Y=x; nothing), Y)
            remotecall(i, x->(global const glrm=x; nothing), glrm)
            remotecall(i, x->(global const alpha=x; nothing), alpha)
            remotecall(i, x->(global const obj=x; nothing), obj)
        end
    end
    @everywhere begin
        # rename data to be easier to access on local proc
        A = LowRankModels.A # since these are shared arrays, this should only be passing the references to them
        X = LowRankModels.X
        Y = LowRankModels.Y
        glrm = LowRankModels.glrm
        k = LowRankModels.glrm.k 
        losses = LowRankModels.glrm.losses
        rx = LowRankModels.glrm.rx
        ry = LowRankModels.glrm.ry # this is a normal array that is getting copied by value to all the procs
        of = LowRankModels.glrm.observed_features
        oe = LowRankModels.glrm.observed_examples
        alpha = LowRankModels.alpha
        obj = LowRankModels.obj
        axpy! = Base.BLAS.axpy!
        gemm! = Base.BLAS.gemm!
        prox! = LowRankModels.prox!
        localcols = LowRankModels.localcols
        grad = LowRankModels.grad
        evaluate = LowRankModels.evaluate

        # cache views into X and Y and the indeces of local columns of X and Y
        m,n = size(A)
        ve = LowRankModels.ContiguousView{Float64,1,Array{Float64,2}}[LowRankModels.view(X.s,:,e) for e=1:m]
        vf = LowRankModels.ContiguousView{Float64,1,Array{Float64,2}}[LowRankModels.view(Y.s,:,f) for f=1:n]
        xlcols = localcols(X)
        ylcols = localcols(Y)
        XY_x = Array(Float64,(length(xlcols), n))
        gemm!('T','N', 1.0, X[:,xlcols], Y.s, 0.0, XY_x) # initialize this for the first iteration
        XY_y = Array(Float64,(m, length(ylcols)))

        # initialize gradient
        g = zeros(k)
    end

    # stopping criterion: stop when decrease in objective < tol
    tol = params.abs_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("fitting GLRM") end
    update_ch!(ch, 0, objective(glrm,X,Y))
    t = time()
    steps_in_a_row = 0

    for i=1:params.max_iter
        @everywhere begin
            # X update
#            XY_x = X[:,xlcols]' * Y  #(rows of the approximation matrix that this processor is in charge of)
            # this is computed before the first iteration and subsequently in the objective evaluation
            for e=xlcols
                scale!(g, 0)  # reset gradient to 0
                # compute gradient of L with respect to Xᵢ as follows:
                # ∇{Xᵢ}L = Σⱼ dLⱼ(XᵢYⱼ)/dXᵢ
                for f in of[e]
                    # but we have no function dLⱼ/dXᵢ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                    # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Yⱼ, where dLⱼ/du is the our grad() function
                    axpy!(grad(losses[f], XY_x[e-xlcols[1]+1,f], A[e,f]), vf[f], g)
                end
                # take a proximal gradient step
                l = length(of[e]) + 1
                scale!(g, -alpha[1]/l)
                ## gradient step: Xᵢ += -(α/l) * ∇{Xᵢ}L
                axpy!(1, g, ve[e])
                ## prox step: Xᵢ = prox_rx(Xᵢ, α/l)
                prox!(rx, ve[e], alpha[1]/l)
            end
        end
        @everywhere begin
            # Y update
            # XY_y = X'*Y[:,ylcols]  (columns of the approximation matrix that this processor is in charge of)
            gemm!('T','N', 1.0, X.s, Y[:,ylcols], 0.0, XY_y)
            for f=ylcols
                scale!(g, 0) # reset gradient to 0
                # compute gradient of L with respect to Yⱼ as follows:
                # ∇{Yⱼ}L = Σⱼ dLⱼ(XᵢYⱼ)/dYⱼ 
                for e in oe[f]
                    # but we have no function dLⱼ/dYⱼ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                    # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Xᵢ, where dLⱼ/du is our grad() function
                    axpy!(grad(losses[f], XY_y[e,f-ylcols[1]+1], A[e,f]), ve[e], g)
                end
                # take a proximal gradient step
                l = length(oe[f]) + 1
                scale!(g, -alpha[1]/l)
                ## gradient step: Yⱼ += -(α/l) * ∇{Yⱼ}L
                axpy!(1, g, vf[f]) 
                ## prox step: Yⱼ = prox_ryⱼ(Yⱼ, α/l)
                prox!(ry[f], vf[f], alpha[1]/l)
            end
        end
        # evaluate objective, splitting up among processes by columns of X
        @everywhere begin
            pid = myid()
            gemm!('T','N', 1.0, X[:,xlcols], Y.s, 0.0, XY_x) # XY = X[:,xlcols]'*Y
            err = 0
            for e=xlcols
                for f in of[e]
                    err += evaluate(losses[f], XY_x[e-xlcols[1]+1,f], A[e,f])
                end
            end
            # add regularization penalty
            for e=xlcols
                err += evaluate(rx,ve[e])
            end
            for f=ylcols
                err += evaluate(ry[f],vf[f]::AbstractArray)
            end
            obj[pid] = err
        end
        # make sure parallel obj eval is the same as local (it is)
        # println("local objective = $(objective(glrm,X,Y)) while shared objective = $(obj[1])")
        # record the best X and Y yet found
        totalobj = sum(obj)
        if totalobj < ch.objective[end]
            t = time() - t
            update_ch!(ch, t, totalobj)
            #copy!(glrm.X, X); copy!(glrm.Y, Y)
            @everywhere begin
                @inbounds for i in localindexes(X)
                    glrm.X[i] = X[i]
                end
                @inbounds for i in localindexes(Y)
                    glrm.Y[i] = Y[i]
                end
            end
            alpha[1] = alpha[1] * 1.05 # sketchy constant
            steps_in_a_row = max(1, steps_in_a_row+1)
            t = time()
        else
            # if the objective went up, reduce the step size, and undo the step
            alpha[1] = alpha[1] * (1 / max(1.5, -steps_in_a_row)) # another sketchy constant
            println("objective went up to $(obj[1]); changing step size to $(alpha[1])")
            @everywhere begin
                @inbounds for i in localindexes(X)
                    X[i] = glrm.X[i]
                end
                @inbounds for i in localindexes(Y)
                    Y[i] = glrm.Y[i]
                end
            end
            @everywhere begin
                gemm!('T','N', 1.0, glrm.X[:,xlcols], glrm.Y.s, 0.0, XY_x) # reset XY to previous best
            end
            # X[:], Y[:] = copy(glrm.X), copy(glrm.Y)
            steps_in_a_row = min(0, steps_in_a_row-1)
        end
        # check stopping criterion
        if i>10 && (steps_in_a_row > 3 && ch.objective[end-1] - totalobj < tol) || alpha[1] <= params.min_stepsize
            break
        end
        if verbose && i%10==0 
            println("Iteration $i: objective value = $(ch.objective[end])") 
        end
    end
    t = time() - t
    update_ch!(ch, t, ch.objective[end])

    return glrm.X', glrm.Y, ch
end