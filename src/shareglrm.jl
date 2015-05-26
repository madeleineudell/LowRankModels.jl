import Base: size, axpy!
import Base.LinAlg: scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView
import Base: shmem_rand, shmem_randn#, acontrank
export GLRM, objective, Params, getindex, display, size, fit, fit!, localcols#, acontrank

# functions for shared arrays
function localcols(Y::SharedArray)
    idxs=localindexes(Y)
    s,t=localindexes(Y)[1],localindexes(Y)[end]
    m,n=size(Y)
    return round(floor((s-1)/m+1)):round(floor(t/m))
end
# acontrank(s::SharedArray,i::Any,c::Any) = acontrank(s.s,i,c)

type GLRM
    A
    observed_features
    observed_examples
    losses::Array{Loss,1}
    rx::Regularizer
    ry::Array{Regularizer,1}
    k::Int64
    X::AbstractArray # k x n
    Y::AbstractArray # k x m
end
# default initializations for obs, X, Y, regularizing every column equally
function GLRM(A,observed_features,observed_examples,losses,rx,ry::Regularizer,k,X,Y)
    rys = Regularizer[typeof(ry)() for i=1:length(losses)]
    for iry in rys
        scale!(iry, scale(ry))
    end
    return GLRM(A,observed_features,observed_examples,losses,rx,rys,k,X,Y)
end
# default initializations for obs, X, and Y
GLRM(A,observed_features,observed_examples,losses,rx,ry,k) = 
    GLRM(A,observed_features,observed_examples,losses,rx,ry,k,shmem_randn(k,size(A,1)),shmem_randn(k,size(A,2)))
GLRM(A,obs,losses,rx,ry,k,X,Y) = 
    GLRM(A,sort_observations(obs,size(A)...)...,losses,rx,ry,k,X,Y)
GLRM(A,obs,losses,rx,ry,k) = 
    GLRM(A,obs,losses,rx,ry,k,shmem_randn(k,size(A,1)),shmem_randn(k,size(A,2)))
function GLRM(A,losses,rx,ry,k)
    m,n = size(A)
    return GLRM(A,fill(1:n, m),fill(1:m, n),losses,rx,ry,k)
end    
function objective(glrm::GLRM,X::Array,Y::Array,Z=nothing; include_regularization=true)
    m,n = size(glrm.A)
    err = 0
    # compute value of loss function
    if Z==nothing Z = X'*Y end
    for i=1:m
        for j in glrm.observed_features[i]
            err += evaluate(glrm.losses[j], Z[i,j], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        for i=1:m
            err += evaluate(glrm.rx,view(X,:,i))
        end
        for j=1:n
            err += evaluate(glrm.ry[j],view(Y,:,j))
        end
    end
    return err
end
objective(glrm::GLRM) = objective(glrm,glrm.X,glrm.Y)
objective(glrm::GLRM,X::SharedArray,Y::SharedArray,args...;kwargs...) = objective(glrm,X.s,Y.s,args...;kwargs...)

type Params
    stepsize # stepsize
    max_iter # maximum number of iterations
    convergence_tol # stop when decrease in objective per iteration is less than convergence_tol*length(obs)
    min_stepsize # use a decreasing stepsize, stop when reaches min_stepsize
end
Params(stepsize,max_iter,convergence_tol) = Params(stepsize,max_iter,convergence_tol,stepsize)
Params() = Params(1,100,.00001,.01)

function sort_observations(obs,m,n; check_empty=false)
    observed_features = Array{Int32,1}[Int32[] for i=1:m]
    observed_examples = Array{Int32,1}[Int32[] for j=1:n]
    for (i,j) in obs
        push!(observed_features[i],j)
        push!(observed_examples[j],i)
    end
    if check_empty && (any(map(x->length(x)==0,observed_examples)) || 
            any(map(x->length(x)==0,observed_features)))
        error("Every row and column must contain at least one observation")
    end
    return observed_features, observed_examples
end

function fit!(glrm::GLRM; params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"),verbose=true)
	
	### initialization (mostly name shortening)
    isa(glrm.A, SharedArray) ? A = glrm.A : A = convert(SharedArray,glrm.A)
    # make sure that we've oriented the factors as shareglrm expects
    if size(glrm.Y,1)!==size(glrm.X,1)
        glrm.X = glrm.X'
    end
	# at any time, glrm.X and glrm.Y will be the best model yet found, while
	# X and Y will be the working variables
    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(glrm.Y) == 0 
        glrm.Y = .1*shmem_randn(size(glrm.Y)...) 
    end
    if isa(glrm.X, SharedArray)
        X, glrm.X = glrm.X, copy(glrm.X)
    else
        X, glrm.X = convert(SharedArray,glrm.X), convert(SharedArray,glrm.X)
    end
    if isa(glrm.Y, SharedArray)
        Y, glrm.Y = glrm.Y, copy(glrm.Y)
    else
        Y, glrm.Y = convert(SharedArray,glrm.Y), convert(SharedArray,glrm.Y)
    end

    ## a few scalars that need to be shared among all processes
    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = Base.shmem_fill(float(params.stepsize),(1,1))
    obj = Base.shmem_fill(0.0,(1,1))

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
        A = LowRankModels.A
        X = LowRankModels.X
        Y = LowRankModels.Y
        glrm = LowRankModels.glrm
        k = LowRankModels.glrm.k
        losses = LowRankModels.glrm.losses
        rx = LowRankModels.glrm.rx
        ry = LowRankModels.glrm.ry
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

        # cache views and local columns
        m,n = size(A)
        ve = LowRankModels.ContiguousView{Float64,1,Array{Float64,2}}[LowRankModels.view(X.s,:,e) for e=1:m]
        vf = LowRankModels.ContiguousView{Float64,1,Array{Float64,2}}[LowRankModels.view(Y.s,:,f) for f=1:n]
        xlcols = localcols(X)
        ylcols = localcols(Y)
        XYX = Array(Float64,(length(xlcols), n))
        XYY = Array(Float64,(m, length(ylcols)))

        # initialize gradient
        g = zeros(k)
    end

    # stopping criterion: stop when decrease in objective < tol
    tol = params.convergence_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("fitting GLRM") end
    update!(ch, 0, objective(glrm,X,Y))
    t = time()
    steps_in_a_row = 0

    for i=1:params.max_iter
        @everywhere begin
            # X update
            gemm!('T','N',1.0,X[:,xlcols],Y.s,0.0,XYX)
            for e=xlcols
                # a gradient of L wrt e
                scale!(g, 0)
                for f in of[e]
                	axpy!(grad(losses[f],XYX[e-xlcols[1]+1,f],A[e,f]), vf[f], g)
                end
                # take a proximal gradient step
                ## gradient step: g = X[e,:] - alpha/l*g
                l = length(of[e]) + 1
                scale!(g, -alpha[1]/l)
                axpy!(1,g,ve[e])
                ## prox step: X[e,:] = prox(g)
                prox!(rx,ve[e],alpha[1]/l)
            end
        end
        @everywhere begin
            # Y update
            # XYY = X'*Y[:,ylcols]
            gemm!('T','N',1.0,X.s,Y[:,ylcols],0.0,XYY)
            for f=ylcols
                # a gradient of L wrt e
                scale!(g, 0)
                for e in oe[f]
                    axpy!(grad(losses[f],XYY[e,f-ylcols[1]+1],A[e,f]), ve[e], g)
                end
                # take a proximal gradient step
                ## gradient step: g = X[e,:] - alpha/l*g
                l = length(oe[f]) + 1
                scale!(g, -alpha[1]/l)
                axpy!(1,g,vf[f])
                ## prox step: X[e,:] = prox(g)
                prox!(ry,vf[f],alpha[1]/l)
            end
        end
        # evaluate objective 
        obj[1] = 0
        @everywhere begin
            XY = X[:,xlcols]'*Y
            err = 0
            for e=xlcols
                for f in of[e]
                    err += evaluate(losses[f], XY[e-xlcols[1]+1,f], A[e,f])
                end
            end
            # add regularization penalty
            for e=xlcols
                err += evaluate(rx,ve[e])
            end
            for f=ylcols
                err += evaluate(ry,vf[f])
            end
            obj[1] = obj[1] + err
        end
        #obj[1] = objective(glrm,X,Y)
        # make sure parallel obj eval is the same as local (it is)
        # println("local objective = $(objective(glrm,X,Y)) while shared objective = $(obj[1])")
        # record the best X and Y yet found
        if obj[1] < ch.objective[end]
            t = time() - t
            update!(ch, t, obj[1])
            #copy!(glrm.X, X); copy!(glrm.Y, Y)
            @everywhere begin
                @inbounds for i in localindexes(X)
                    glrm.X[i] = X[i]
                end
                @inbounds for i in localindexes(Y)
                    glrm.Y[i] = Y[i]
                end
            end
            alpha[1] = alpha[1] * 1.05
            steps_in_a_row = max(1, steps_in_a_row+1)
            t = time()
        else
            # if the objective went up, reduce the step size, and undo the step
            alpha[1] = alpha[1] * (1 / max(1.5, -steps_in_a_row))
            @everywhere begin
                @inbounds for i in localindexes(X)
                    X[i] = glrm.X[i]
                end
                @inbounds for i in localindexes(Y)
                    Y[i] = glrm.Y[i]
                end
            end
            steps_in_a_row = min(0, steps_in_a_row-1)
        end
        # check stopping criterion
        if i>10 && (steps_in_a_row > 3 && ch.objective[end-1] - obj[1] < tol) || alpha[1] <= params.min_stepsize
            break
        end
        if verbose && i%10==0 
            println("Iteration $i: objective value = $(ch.objective[end])") 
        end
    end
    t = time() - t
    update!(ch, t, ch.objective[end])

    return glrm.X,glrm.Y,ch
end

function fit(glrm::GLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fit!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X,Y,ch
end