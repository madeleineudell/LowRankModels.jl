import Base: size, axpy!
import Base.LinAlg.scale!
import Base.BLAS: gemm!
import ArrayViews: view, StridedView, ContiguousView

export GLRM, objective, Params, getindex, display, size, fit!, fit

ObsArray = Union(Array{Array{Int,1},1}, Array{UnitRange{Int},1})

### GLRM TYPE
type GLRM
    A::Array{Float64,2}          # The data table transformed into a coded array 
    losses::Array{Loss,1}        # array of loss functions
    rx::Regularizer              # The regularization to be applied to each row of Xᵀ (column of X)
    ry::Array{Regularizer,1}     # Array of regularizers to be applied to each column of Y
    k::Int64                     # Desired rank 
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed  
    X::Array{Float64,2}          # Representation of data in low-rank space. A ≈ X'Y
    Y::Array{Float64,2}          # Representation of features in low-rank space. A ≈ X'Y
end
function GLRM(A, losses, rx, ry, k; 
              X = randn(k,size(A,1)), Y = randn(k,size(A,2)),
              obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
              observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
              observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
              offset = true, scale = true)
    # if isa(ry, Regularizer)
    # 	# println("single reg given, converting")
        # 	ry = fill(ry, size(losses))
    # end
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
GLRM(A, losses, rx, ry::Regularizer, k; kwargs...) = GLRM(A, losses, rx, fill(ry,size(losses)), k; kwargs...)


### OBSERVATION TUPLES TO ARRAYS
function sort_observations(obs::Array{(Int,Int),1}, m::Int, n::Int; check_empty=false)
    observed_features = Array{Int32,1}[Int32[] for i=1:m]
    observed_examples = Array{Int32,1}[Int32[] for j=1:n]
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

### OBJECTIVE FUNCTION EVALUATION
function objective(glrm::GLRM, X, Y, XY=nothing; include_regularization=true)
    m,n = size(glrm.A)
    err = 0
    # compute value of loss function
    if XY == nothing # if the user is calling this independent of the optimization,
    	XY = Array(Float64, (m,n))	# we need to compute XY for her
    	gemm!('T','N',1.0,X,Y,0.0,XY) 
    end
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
objective(glrm::GLRM, args...; kwargs...) = 
    objective(glrm, glrm.X, glrm.Y, args...; kwargs...)

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

### FITTING
function fit!(glrm::GLRM; params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"),verbose=true)
	
	### initialization
	A = glrm.A # rename these for easier local access
	losses = glrm.losses
	rx = glrm.rx
	ry = glrm.ry
	# at any time, glrm.X and glrm.Y will be the best model yet found, while
	# X and Y will be the working variables
	X = copy(glrm.X); Y = copy(glrm.Y)
	k = glrm.k

	m,n = size(A)
	XY = Array(Float64, (m, n))
	gemm!('T','N',1.0,X,Y,0.0,XY) # XY = X' * Y initial calculation

    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0 
    	Y = .1*randn(k,n) 
    end

    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = params.stepsize
    # stopping criterion: stop when decrease in objective < tol
    tol = params.convergence_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    update!(ch, 0, objective(glrm))
    t = time()
    steps_in_a_row = 0
    g = zeros(k)

    # cache views
    ve = ContiguousView{Float64,1,Array{Float64,2}}[view(X,:,e) for e=1:m]
    vf = ContiguousView{Float64,1,Array{Float64,2}}[view(Y,:,f) for f=1:n]

    for i=1:params.max_iter
# STEP 1: X update
        # XY = X' * Y this is computed before the first iteration and subsequently in the objective evaluation
        for e=1:m # doing this means looping over XY in row-major order, but otherwise we couldn't parallelize over Xᵢs
            scale!(g, 0)# reset gradient to 0
            # compute gradient of L with respect to Xᵢ as follows:
            # ∇{Xᵢ}L = Σⱼ dLⱼ(XᵢYⱼ)/dXᵢ
            for f in glrm.observed_features[e]
                # but we have no function dLⱼ/dXᵢ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Yⱼ, where dLⱼ/du is our grad() function
                axpy!(grad(losses[f],XY[e,f],A[e,f]), vf[f], g)
            end
            # take a proximal gradient step
            l = length(glrm.observed_features[e]) + 1
            scale!(g, -alpha/l)
            ## gradient step: Xᵢ += -(α/l) * ∇{Xᵢ}L
            axpy!(1,g,ve[e])
            ## prox step: Xᵢ = prox_rx(Xᵢ, α/l)
            prox!(rx,ve[e],alpha/l)
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new X
# STEP 2: Y update
        for f=1:n
            scale!(g, 0) # reset gradient to 0
            # compute gradient of L with respect to Yⱼ as follows:
            # ∇{Yⱼ}L = Σⱼ dLⱼ(XᵢYⱼ)/dYⱼ 
            for e in glrm.observed_examples[f]
                # but we have no function dLⱼ/dYⱼ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Xᵢ, where dLⱼ/du is our grad() function
            	axpy!(grad(losses[f],XY[e,f],A[e,f]), ve[e], g)
            end
            # take a proximal gradient step
            l = length(glrm.observed_examples[f]) + 1
            scale!(g, -alpha/l)
            ## gradient step: Yⱼ += -(α/l) * ∇{Yⱼ}L
            axpy!(1,g,vf[f]) 
            ## prox step: Yⱼ = prox_ryⱼ(Yⱼ, α/l)
            prox!(ry[f],vf[f],alpha/l)
        end
        gemm!('T','N',1.0,X,Y,0.0,XY) # Recalculate XY using the new Y
# STEP 3: Check objective
        obj = objective(glrm, X, Y, XY) 
        # record the best X and Y yet found
        if obj < ch.objective[end]
            t = time() - t
            update!(ch, t, obj)
            copy!(glrm.X, X); copy!(glrm.Y, Y)
            alpha = alpha * 1.05
            steps_in_a_row = max(1, steps_in_a_row+1)
            t = time()
        else
            # if the objective went up, reduce the step size, and undo the step
            alpha = alpha / max(1.5, -steps_in_a_row)
            if verbose println("obj went up to $obj; reducing step size to $alpha") end
            copy!(X, glrm.X); copy!(Y, glrm.Y)
            steps_in_a_row = min(0, steps_in_a_row-1)
            gemm!('T','N',1.0,X,Y,0.0,XY) # Revert back to the old XY (previous best)
        end
# STEP 4: Check stopping criterion
        if i>10 && (steps_in_a_row > 3 && ch.objective[end-1] - obj < tol) || alpha <= params.min_stepsize
            break
        end
        if verbose && i%10==0 
            println("Iteration $i: objective value = $(ch.objective[end])") 
        end
    end
    t = time() - t
    update!(ch, t, ch.objective[end])

    return glrm.X, glrm.Y, ch
end

function fit(glrm::GLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fit!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X',Y,ch
end