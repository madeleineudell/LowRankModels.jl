### Proximal gradient method
export SparseProxGradParams, fit!

mutable struct SparseProxGradParams<:AbstractParams
    stepsize::Float64 # initial stepsize
    max_iter::Int # maximum number of outer iterations
    inner_iter::Int # how many prox grad steps to take on X before moving on to Y (and vice versa)
    abs_tol::Float64 # stop if objective decrease upon one outer iteration is less than this
    min_stepsize::Float64 # use a decreasing stepsize, stop when reaches min_stepsize
end
function SparseProxGradParams(stepsize::Number=1.0; # initial stepsize
				              max_iter::Int=100, # maximum number of outer iterations
				              inner_iter::Int=1, # how many prox grad steps to take on X before moving on to Y (and vice versa)
				              abs_tol::Float64=0.00001, # stop if objective decrease upon one outer iteration is less than this
				              min_stepsize::Float64=0.01*stepsize) # stop if stepsize gets this small
    stepsize = convert(Float64, stepsize)
    return SparseProxGradParams(stepsize, max_iter, inner_iter, abs_tol, min_stepsize)
end

### FITTING
function fit!(glrm::GLRM, params::SparseProxGradParams;
			  ch::ConvergenceHistory=ConvergenceHistory("SparseProxGradGLRM"),
			  verbose=true,
			  kwargs...)
	println(params)
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

    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0
    	Y = .1*randn(k,n)
    end

    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = params.stepsize
    # stopping criterion: stop when decrease in objective < tol
    tol = params.abs_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    update_ch!(ch, 0, objective(glrm; sparse=true))
    t = time()
    steps_in_a_row = 0
    g = zeros(k)

    # cache views
    ve = [view(X,:,e) for e=1:m]
    vf = [view(Y,:,f) for f=1:n]

    for i=1:params.max_iter
# STEP 1: X update
        for inneri=1:params.inner_iter
        for e=1:m # doing this means looping over XY in row-major order, but otherwise we couldn't parallelize over Xᵢs
            rmul!(g, 0)# reset gradient to 0
            # compute gradient of L with respect to Xᵢ as follows:
            # ∇{Xᵢ}L = Σⱼ dLⱼ(XᵢYⱼ)/dXᵢ
            for f in glrm.observed_features[e]
                # but we have no function dLⱼ/dXᵢ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ (dLⱼ(XᵢYⱼ)/du * Yⱼ), where dLⱼ/du is our grad() function
                # our estimate for A[e,f] is given by dot(ve[e],vf[f])
                axpy!(grad(losses[f],dot(ve[e],vf[f]),A[e,f]), vf[f], g)
            end
            # take a proximal gradient step
            l = length(glrm.observed_features[e]) + 1
            rmul!(g, -alpha/l)
            ## gradient step: Xᵢ += -(α/l) * ∇{Xᵢ}L
            axpy!(1,g,ve[e])
            ## prox step: Xᵢ = prox_rx(Xᵢ, α/l)
            prox!(rx[e],ve[e],alpha/l)
        end
    	end
# STEP 2: Y update
        for inneri=1:params.inner_iter
        for f=1:n
            rmul!(g, 0) # reset gradient to 0
            # compute gradient of L with respect to Yⱼ as follows:
            # ∇{Yⱼ}L = Σⱼ dLⱼ(XᵢYⱼ)/dYⱼ
            for e in glrm.observed_examples[f]
                # but we have no function dLⱼ/dYⱼ, only dLⱼ/d(XᵢYⱼ) aka dLⱼ/du
                # by chain rule, the result is: Σⱼ dLⱼ(XᵢYⱼ)/du * Xᵢ, where dLⱼ/du is our grad() function
            	axpy!(grad(losses[f],dot(ve[e],vf[f]),A[e,f]), ve[e], g)
            end
            # take a proximal gradient step
            l = length(glrm.observed_examples[f]) + 1
            rmul!(g, -alpha/l)
            ## gradient step: Yⱼ += -(α/l) * ∇{Yⱼ}L
            axpy!(1,g,vf[f])
            ## prox step: Yⱼ = prox_ryⱼ(Yⱼ, α/l)
            prox!(ry[f],vf[f],alpha/l)
        end
        end
# STEP 3: Check objective
        obj = objective(glrm, X, Y; sparse=true)
        # record the best X and Y yet found
        if obj < ch.objective[end]
            t = time() - t
            update_ch!(ch, t, obj)
            copy!(glrm.X, X); copy!(glrm.Y, Y) # save new best X and Y
            alpha = alpha * 1.05
            steps_in_a_row = max(1, steps_in_a_row+1)
            t = time()
        else
            # if the objective went up, reduce the step size, and undo the step
            alpha = alpha / max(1.5, -steps_in_a_row)
            if verbose println("obj went up to $obj; reducing step size to $alpha") end
            copy!(X, glrm.X); copy!(Y, glrm.Y) # revert back to last X and Y
            steps_in_a_row = min(0, steps_in_a_row-1)
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
    update_ch!(ch, t, ch.objective[end])

    return glrm.X, glrm.Y, ch
end
