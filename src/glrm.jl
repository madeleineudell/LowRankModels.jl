import Base: size, axpy!
import Base.LinAlg.scale!
import ArrayViews: view, StridedView, ContiguousView

export GLRM, objective, Params, getindex, display, size, fit, fit!

type GLRM
    A
    observed_features
    observed_examples
    losses::Array{Loss,1}
    rx::Regularizer
    ry::Array{Regularizer,1}
    k::Int64
    X::Array{Float64,2}
    Y::Array{Float64,2}
end
# default initializations for obs, X, Y, regularizing every column equally
function GLRM(A,observed_features,observed_examples,losses,rx,ry::Regularizer,k,X,Y)
    rys = Regularizer[typeof(ry)() for i=1:length(losses)]
    for iry in rys
        scale!(iry, scale(ry))
    end
    return GLRM(A,observed_features,observed_examples,losses,rx,rys,k,X,Y)
end
GLRM(A,observed_features,observed_examples,losses,rx,ry,k) = 
    GLRM(A,observed_features,observed_examples,losses,rx,ry,k,randn(size(A,1),k),randn(k,size(A,2)))
GLRM(A,obs,losses,rx,ry,k,X,Y) = 
    GLRM(A,sort_observations(obs,size(A)...)...,losses,rx,ry,k,X,Y)
GLRM(A,obs,losses,rx,ry,k) = 
    GLRM(A,obs,losses,rx,ry,k,randn(size(A,1),k),randn(k,size(A,2)))
function GLRM(A,losses,rx,ry,k)
    m,n = size(A)
    return GLRM(A,fill(1:n, m),fill(1:m, n),losses,rx,ry,k)
end 
function objective(glrm::GLRM,X,Y,Z=nothing; include_regularization=true)
    m,n = size(glrm.A)
    err = 0
    # compute value of loss function
    if Z==nothing Z = X*Y end
    for i=1:m
        for j in glrm.observed_features[i]
            err += evaluate(glrm.losses[j], Z[i,j], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        for i=1:m
            err += evaluate(glrm.rx,view(X,i,:))
        end
        for j=1:n
            err += evaluate(glrm.ry[j],view(Y,:,j))
        end
    end
    return err
end
objective(glrm::GLRM, args...; kwargs...) = 
    objective(glrm, glrm.X, glrm.Y, args...; kwargs...)

type Params
    stepsize # stepsize
    max_iter # maximum number of iterations
    convergence_tol # stop when decrease in objective per iteration is less than convergence_tol*length(obs)
    min_stepsize # use a decreasing stepsize, stop when reaches min_stepsize
end
Params(stepsize,max_iter,convergence_tol) = Params(stepsize,max_iter,convergence_tol,.01*stepsize)
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
    
    ### initialization
    A = glrm.A
    m,n = size(A)
    losses = glrm.losses
    rx = glrm.rx
    ry = glrm.ry
    # at any time, glrm.X and glrm.Y will be the best model yet found, while
    # X and Y will be the working variables
    X = copy(glrm.X); Y = copy(glrm.Y)
    k = glrm.k

    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0 
        Y = .1*randn(k,n) 
    end

    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = params.stepsize
    alpharow = fill(float(params.stepsize),m)
    alphacol = fill(float(params.stepsize),n)
    objbyrow = zeros(m)
    objbycol = zeros(n)
    # stopping criterion: stop when decrease in objective < tol
    tol = params.convergence_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    update!(ch, 0, objective(glrm))
    t = time()
    steps_in_a_row = 0
    g = zeros(k)

    # cache views
    ve = StridedView{Float64,2,0,Array{Float64,2}}[view(X,e,:) for e=1:m]
    vf = ContiguousView{Float64,1,Array{Float64,2}}[view(Y,:,f) for f=1:n]

    for i=1:params.max_iter
        # X update
        XY = X*Y
        for e=1:m
            # calculate a gradient of L wrt e, and the objective value for this row
            scale!(g, 0)
            objbyrow[e] = evaluate(rx,ve[e])
            for f in glrm.observed_features[e]
                axpy!(grad(losses[f],XY[e,f],A[e,f]), vf[f], g)
                objbyrow[e] += evaluate(losses[f], XY[e,f], A[e,f])
            end
            # take a proximal gradient step
            ## gradient step: g = X[e,:] - alpha/l*g
            l = length(glrm.observed_features[e]) + 1
            scale!(g, -alpharow[e]/l)
            axpy!(1,g,ve[e])
            ## prox step: X[e,:] = prox(g)
            prox!(rx,ve[e],alpharow[e]/l)
            
            # see if solution improved
            xy = ve[e]*Y
            err = evaluate(rx,ve[e])
            for f in glrm.observed_features[e]
                err += evaluate(losses[f], xy[f], A[e,f])
            end
            if err > objbyrow[e]
                # println("row $e worsened; undoing step")
                alpharow[e] *= .7
                X[e,:] = glrm.X[e,:]
                # scale!(ve[e],0)
                # axpy!(1,glrm.X[e,:],ve[e])
            else
                alpharow[e] *= 1.05
                glrm.X[e,:] = X[e,:]
                # scale!(glrm.X[e,:],0)
                # axpy!(1,ve[e],glrm.X[e,:])
            end   
        end
        # Y update
        XY = X*Y
        for f=1:n
            # a gradient of L wrt f
            scale!(g, 0)
            objbycol[f] = evaluate(ry[f],vf[f])
            for e in glrm.observed_examples[f]
                axpy!(grad(losses[f],XY[e,f],A[e,f]), ve[e], g)
                objbycol[f] += evaluate(losses[f], XY[e,f], A[e,f])
            end
            # take a proximal gradient step
            ## gradient step: g = Y[:,f] - alpha/l*g
            l = length(glrm.observed_examples[f]) + 1
            scale!(g, -alphacol[f]/l)
            axpy!(1,g,vf[f]) 
            ## prox step: X[e,:] = prox(g)
            prox!(ry[f],vf[f],alphacol[f]/l)

            # see if solution improved
            xy = X*vf[f]
            err = evaluate(ry[f],vf[f])
            for e in glrm.observed_examples[f]
                err += evaluate(losses[f], xy[e], A[e,f])
            end
            if err > objbycol[f]
                #println("col $f worsened; undoing step")
                alphacol[f] *= .7
                Y[:,f] = glrm.Y[:,f]
                # scale!(vf[f],0)
                # axpy!(1,glrm.Y[:,f],vf[f])
            else
                alphacol[f] *= 1.05
                glrm.Y[:,f] = Y[:,f]
                # scale!(glrm.Y[:,f],0)
                # axpy!(1,vf[f],glrm.Y[:,f])
                objbycol[f] = err
            end   
        end
        obj = sum(objbycol)
        t = time() - t
        update!(ch, t, obj)
        t = time()        
        # check sanity
        # @show obj = objective(glrm,X,Y)
        # obj = objective(glrm)
        # # record the best X and Y yet found
        # if obj < ch.objective[end]
        #     t = time() - t
        #     update!(ch, t, obj)
        #     t = time()
        # else
        #     # the objective should never go up
        #     warn("objective went up; why?")
        #     copy!(X, glrm.X); copy!(Y, glrm.Y)
        # end
        # check stopping criterion
        if i>10 && length(ch.objective)>3 && (ch.objective[end-1] - obj < tol || 
                    (median(alpharow) <= params.min_stepsize && 
                        median(alphacol) <= params.min_stepsize))
            break
        # else
        #     println(ch.objective[end-1] - obj," not < ", tol)
        #     println("alpharow")
        #     println("\t", median(alpharow))
        #     println("\t", mean(alpharow))
        #     println("\t", maximum(alpharow))
        #     println("\t", minimum(alpharow))
        #     println("alphacol")
        #     println("\t", median(alphacol))
        #     println("\t", mean(alphacol))
        #     println("\t", maximum(alphacol))
        #     println("\t", minimum(alphacol))        
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