import Base: size, axpy!
import Base.LinAlg: scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView
import Base: shmem_rand, shmem_randn#, acontrank

export GLRM, 
       Params, fit!, fit, localcols,
       objective, error_metric, 
       add_offset!, equilibrate_variance!, fix_latent_features!

# functions for shared arrays
function localcols(Y::SharedArray)
    idxs=localindexes(Y)
    s,t=localindexes(Y)[1],localindexes(Y)[end]
    m,n=size(Y)
    return round(floor((s-1)/m+1)):round(floor(t/m))
end
# acontrank(s::SharedArray,i::Any,c::Any) = acontrank(s.s,i,c)

typealias ObsArray Union(Array{Array{Int,1},1}, Array{UnitRange{Int},1})

### GLRM TYPE
type GLRM
    A::Array{Float64,2}          # The data table transformed into a coded array 
    losses::Array{Loss,1}        # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{Regularizer,1}     # Array of regularizers to be applied to each column of Y
    k::Int                     # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::SharedArray{Float64,2}    # Representation of data in low-rank space. A ≈ X'Y
    Y::SharedArray{Float64,2}    # Representation of features in low-rank space. A ≈ X'Y
end
function GLRM(A::AbstractArray, losses::Array{Loss,1}, rx::Regularizer, ry::Array{Regularizer,1}, k::Int; 
              X = shmem_randn(k,size(A,1)), Y = shmem_randn(k,size(A,2)),
              obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
              observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
              observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
              offset = true, scale = true)
    # Check dimensions of the arguments
    A = convert(Array{Float64,2}, A)
    m,n = size(A)
    if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
    if length(ry)!=n error("There must be either one Y regularizer or as many Y regularizers as there are columns in the data matrix") end
    if size(X)!=(k,m) error("X must be of size (k,m) where m is the number of rows in the data matrix.
                                    This is the transpose of the standard notation used in the paper, but it 
                                    makes for better memory management.") end
    if size(Y)!=(k,n) error("Y must be of size (k,n) where n is the number of columns in the data matrix.") end
    # If the user passed in arrays, make sure to convert them to shared arrays
    if !isa(X, SharedArray) X = convert(SharedArray, X) end
    if !isa(Y, SharedArray) Y = convert(SharedArray, Y) end
    if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
        # println("no obs given, using observed_features and observed_examples")
        glrm = GLRM(A,losses,rx,ry,k, observed_features, observed_examples, X,Y)
    else # otherwise unpack the tuple list into arrays
        # println("unpacking obs into array")
        glrm = GLRM(A,losses,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y)
    end
    if scale # scale losses (and regularizers) so they all have equal variance
        equilibrate_variance!(glrm)
    end
    if offset # don't penalize the offset of the columns
        add_offset!(glrm)
    end
    return glrm
end
function GLRM(A, losses, rx, ry::Regularizer, k; kwargs...)
    ry_array = convert(Array{Regularizer,1}, fill(ry,size(losses)))
    GLRM(A, losses, rx, ry_array, k; kwargs...)
end

### OBSERVATION TUPLES TO ARRAYS
function sort_observations(obs::Array{(Int,Int),1}, m::Int, n::Int; check_empty=false)
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
    glrm.ry = [fixed_latent_features(glrm.ry[i], glrm.Y[1:n,i]) for i in 1:length(glrm.ry)]
    return glrm
end

### OBJECTIVE FUNCTION EVALUATION
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}, 
                   XY::Array{Float64,2}; include_regularization=true)
    m,n = size(glrm.A)
    err = 0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += evaluate(glrm.losses[j], XY[i,j], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        for i=1:m
            err += evaluate(glrm.rx, view(X,:,i))
        end
        for j=1:n
            err += evaluate(glrm.ry[j], view(Y,:,j))
        end
    end
    return err
end
# The user can also pass in X and Y and `objective` will compute XY for them
function objective(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}; kwargs...)
    XY = Array(Float64, size(glrm.A)) 
    gemm!('T','N',1.0,X,Y,0.0,XY) 
    objective(glrm, X, Y, XY; kwargs...)
end
function objective(glrm::GLRM, X::SharedArray{Float64,2}, Y::SharedArray{Float64,2}; kwargs...)
    objective(glrm, convert(Array,X), convert(Array,Y); kwargs...)
end
# Or just the GLRM and `objective` will use glrm.X and .Y
objective(glrm::GLRM; kwargs...) = objective(glrm, glrm.X, glrm.Y; kwargs...)

## ERROR METRIC EVALUATION (BASED ON DOMAINS OF THE DATA)
function error_metric(glrm::GLRM, XY::Array{Float64,2}, domains::Array{Domain,1})
    m,n = size(glrm.A)
    err = 0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += error_metric(domains[j], glrm.losses[j], XY[i,j], glrm.A[i,j])
        end
    end
    return err
end
# The user can also pass in X and Y and `error_metric` will compute XY for them
function error_metric(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}, domains::Array{Domain,1})
    XY = Array(Float64, size(glrm.A)) 
    gemm!('T','N',1.0,X,Y,0.0,XY) 
    error_metric(glrm, XY, domains)
end
function error_metric(glrm::GLRM, X::SharedArray{Float64,2}, 
                      Y::SharedArray{Float64,2}, domains::Array{Domain,1})
    error_metric(glrm, convert(Array,X), convert(Array,Y), domains)
end
# Or just the GLRM and `error_metric` will use glrm.X and .Y
error_metric(glrm::GLRM, domains::Array{Domain,1}) = error_metric(glrm, glrm.X, glrm.Y, domains)
error_metric(glrm::GLRM) = error_metric(glrm, Domain[l.domain for l in glrm.losses])

### PARAMETERS TYPE
type Params
    stepsize # stepsize
    max_iter # maximum number of iterations
    convergence_tol # stop when decrease in objective per iteration is less than convergence_tol*length(obs)
    min_stepsize # use a decreasing stepsize, stop when reaches min_stepsize
end
function Params(stepsize=1; max_iter=100, convergence_tol=0.00001, min_stepsize=0.01*stepsize) 
    return Params(stepsize, max_iter, convergence_tol, min_stepsize)
end

function fit!(glrm::GLRM; params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"),verbose=true)
	
	### initialization (mostly name shortening)
    isa(glrm.A, SharedArray) ? A = glrm.A : A = convert(SharedArray,glrm.A)
    # make sure that we've oriented the factors as shareglrm expects
    if size(glrm.Y,1)!==size(glrm.X,1)
        glrm.X = glrm.X'
    end
    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(glrm.Y) == 0 
        glrm.Y = .1*shmem_randn(size(glrm.Y)...) 
    end
    # at any time, glrm.X and glrm.Y will be the best model yet found, while
    # X and Y will be the working variables. All of these are shared arrays.
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
    tol = params.convergence_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("fitting GLRM") end
    update!(ch, 0, objective(glrm,X,Y))
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
        totalobj = sum(obj)
        # make sure parallel obj eval is the same as local (it is)
        # println("local objective = $(objective(glrm)) while shared objective = $(totalobj)")
        # record the best X and Y yet found
        if totalobj < ch.objective[end]
            t = time() - t
            update!(ch, t, totalobj)
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
    update!(ch, t, ch.objective[end])

    return glrm.X.s, glrm.Y.s, ch
end

function fit(glrm::GLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fit!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X',Y,ch
end