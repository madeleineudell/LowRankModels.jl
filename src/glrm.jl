import Stats.sample, Base.size

export GLRM, objective, Params, FunctionArray, getindex, display, size, fit, fit!

type GLRM
	A
	observed_features
	observed_examples
	losses::Array{Loss, }
	rx::Regularizer
	ry::Regularizer
	k::Int64
	X::Array{Float64,2}
	Y::Array{Float64,2}
end
# default initializations for obs, X, and Y
GLRM(A,obs,losses,rx,ry,k,X,Y) = 
	GLRM(A,sort_observations(obs,size(A)...)...,losses,rx,ry,k,X,Y)
GLRM(A,obs,losses,rx,ry,k) = 
	GLRM(A,obs,losses,rx,ry,k,randn(size(A,1),k),randn(k,size(A,2)))
GLRM(A,losses,rx,ry,k) = 
	GLRM(A,[(i,j) for i=1:size(A,1),j=1:size(A,2)][:],losses,rx,ry,k)	
function objective(glrm::GLRM,X,Y; include_regularization=true)
	m,n = size(glrm.A)
	err = 0
	# compute value of loss function
	Z = X * Y
	for i=1:n
		for j in glrm.observed_features[i]
			err += evaluate(glrm.losses[j], Z[i,j], glrm.A[i,j])
		end
	end
	# add regularization penalty
	if include_regularization
		for i=1:m
			err += evaluate(glrm.rx,X[i,:])
		end
		for j=1:n
			err += evaluate(glrm.ry,Y[:,j])
		end
	end
	return err
end
objective(glrm::GLRM) = objective(glrm,glrm.X,glrm.Y)

type Params
	stepsize # stepsize
	max_iter # maximum number of iterations
	convergence_tol # stop when decrease in objective per iteration is less than convergence_tol*length(obs)
	min_stepsize # use a decreasing stepsize, stop when reaches min_stepsize
end
Params(stepsize,max_iter,convergence_tol) = Params(stepsize,max_iter,convergence_tol,stepsize)
Params() = Params(1,100,.001)

type FunctionArray<:AbstractArray
	f::Function
	arr::Array
end
getindex(fa::FunctionArray,idx::Integer...) = x->fa.f(x,fa.arr[idx...])
display(fa::FunctionArray) = println("FunctionArray($(fa.f),$(fa.arr))")
size(fa::FunctionArray) = size(fa.arr)

type ColumnFunctionArray<:AbstractArray
    f::Array{Function,1}
    arr::AbstractArray
end
getindex(fa::ColumnFunctionArray,idx::Integer...) = x->fa.f[idx[2]](x,fa.arr[idx...])
display(fa::ColumnFunctionArray) = println("FunctionArray($(fa.f),$(fa.arr))")
size(fa::ColumnFunctionArray) = size(fa.arr)

function sort_observations(obs,m,n)
    observed_features = [Int32[] for i=1:m]
    observed_examples = [Int32[] for j=1:n]
    for (i,j) in obs
        push!(observed_features[i],j)
        push!(observed_examples[j],i)
    end
    if any(map(x->length(x)==0,observed_examples)) || 
        	any(map(x->length(x)==0,observed_features))
        error("Every row and column must contain at least one observation")
    end
    return observed_features, observed_examples
end

function fit(glrm::GLRM,params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"); verbose=true)
	
	# initialization
	gradL = ColumnFunctionArray(map(grad,glrm.losses),glrm.A)
	m,n = size(gradL)
	X, Y = copy(glrm.X), copy(glrm.Y)
	k = glrm.k

	# scale optimization parameters
	## stopping criterion: stop when decrease in objective < tol
	tol = params.convergence_tol * sum(map(length,glrm.observed_features))
	## ensure step size alpha = O(1/g) is apx bounded by maximum size of gradient
	N = maximum(map(length,glrm.observed_features))
	M = maximum(map(length,glrm.observed_examples))
	alpha = params.stepsize / max(M,N)

	# alternating updates of X and Y
	if verbose println("Fitting GLRM") end
	update!(ch, 0, objective(glrm))
	t = time()
	for i=1:params.max_iter
		# X update
		XY = X*Y
		for e=1:m
			# a gradient of L wrt e
			g = zeros(1,k)
			for f in glrm.observed_features[e]
				g += gradL[e,f](XY[e,f])*Y[:,f:f]'
			end
			# take a proximal gradient step
			X[e,:] = prox(glrm.rx)(X[e:e,:]-alpha*g,alpha)
		end
		# Y update
		XY = X*Y
		for f=1:n
			# a gradient of L wrt f
			g = zeros(k,1)
			for e in glrm.observed_examples[f]
				g += X[e:e,:]'*gradL[e,f](XY[e,f])
			end
			# take a proximal gradient step
			Y[:,f] = prox(glrm.ry)(Y[:,f:f]-alpha*g,alpha)
		end
		obj = objective(glrm,X,Y)
		# record the best X and Y yet found
		if obj < ch.objective[end]
			t = time() - t
			update!(ch, t, obj)
			glrm.X[:], glrm.Y[:] = X, Y
			t = time()
		end
		# check stopping criterion
		if i>10 && ch.objective[end-1] - obj < tol
			if alpha <= params.min_stepsize / max(M,N)
				break
			else
				alpha = alpha/2
			end
		end
		if verbose && i%10==0 
			println("Iteration $i: objective value = $(ch.objective[end])") 
		end
	end

	return glrm.X,glrm.Y,ch
end
function fit!(glrm::GLRM,params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"))
	glrm.X, glrm.Y = fit(glrm,params)
end