export cross_validate, cv_by_iter, regularization_path, get_train_and_test, precision_at_k

# the loss function evaluates the objective minus the regularization
# it is the default error metric
loss_fn(args...; kwargs...) = objective(args...; include_regularization=false, kwargs...)

# to use with error_metric when we have domains in the namespace, call as:
# cross_validate(glrm, error_fn = error_metric(glrm,domains,glrm.X,glrm.Y))
function cross_validate(glrm::AbstractGLRM;
                        nfolds=5, params=Params(),
                        verbose=true, use_folds=nfolds,
                        error_fn=loss_fn,
                        init=nothing,
                        do_obs_check = false)
    if verbose println("flattening observations") end
#    obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
    obs = flatten_observations(glrm.observed_features)
    if verbose println("computing CV folds") end
    folds = getfolds(obs, nfolds, size(glrm.A)..., do_check = do_obs_check)
    train_glrms = Array{typeof(glrm)}(nfolds)
    test_glrms = Array{typeof(glrm)}(nfolds)
    train_error = @compat Array{Float64}(nfolds)
    test_error = @compat Array{Float64}(nfolds)
    for ifold=1:use_folds
        if verbose println("\nforming train and test GLRM for fold $ifold") end
        train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = folds[ifold]
	    ntrain = sum(map(length, train_observed_features))
    	ntest = sum(map(length, test_observed_features))
	    if verbose println("training model on $ntrain samples and testing on $ntest") end
        # form glrm on training dataset
        train_glrms[ifold] = copy_estimate(glrm)
        train_glrms[ifold].observed_examples = train_observed_examples
        train_glrms[ifold].observed_features = train_observed_features
        # form glrm on testing dataset
        test_glrms[ifold] = copy_estimate(glrm)
        test_glrms[ifold].observed_examples = test_observed_examples
        test_glrms[ifold].observed_features = test_observed_features
        # evaluate train and test error
        if verbose println("fitting train GLRM for fold $ifold") end
        if init != nothing
            init(train_glrms[ifold])
        end
        fit!(train_glrms[ifold], params, verbose=verbose)
        if verbose println("computing train and test error for fold $ifold:") end
        train_error[ifold] = error_fn(train_glrms[ifold],
            parameter_estimate(train_glrms[ifold])...) / ntrain
        if verbose println("\ttrain error: $(train_error[ifold])") end
        test_error[ifold] = error_fn(test_glrms[ifold],
            parameter_estimate(train_glrms[ifold])...) / ntest
        if verbose println("\ttest error:  $(test_error[ifold])") end
    end
    return train_error, test_error, train_glrms, test_glrms
end

@compat function getfolds(obs::Array{Tuple{Int,Int},1}, nfolds, m, n; ntrials = 5, do_check = true)
    # partition elements of obs into nfolds groups
    groups = @compat Array{Int}(size(obs))
    rand!(groups, 1:nfolds)  # fill an array with random 1 through N
    # create the training and testing observations for each fold
    folds = @compat Array{Tuple}(nfolds)
    for itrial = 1:ntrials
        enough_observations = 0
        for ifold=1:nfolds
            train = obs[filter(i->groups[i]!=ifold, 1:length(obs))] # all the obs that didn't get the ifold label
            train_observed_features, train_observed_examples = sort_observations(train,m,n)
            if !do_check ||
                (check_enough_observations(train_observed_features) && check_enough_observations(train_observed_examples))
                    enough_observations += 1
            else
                warn("Not enough data to cross validate; one of the cross validation folds has no observations in one row or column. Trying again...")
                break
            end
            test = obs[filter(i->groups[i]==ifold, 1:length(obs))] # all the obs that did
            test_observed_features, test_observed_examples = sort_observations(test,m,n,check_empty=false)
            folds[ifold] = (train_observed_features, train_observed_examples,
                        test_observed_features,  test_observed_examples)
        end
        if enough_observations == nfolds
            return folds
        end
    end
    error("Not enough data to cross validate automatically.")
end

function check_enough_observations(observed_examples_or_features)
    all(map(length, observed_examples_or_features) .> 0)
end

function get_train_and_test(obs, m, n, holdout_proportion=.1)

    # generate random uniform number for each observation
    groups = @compat Array{Float64}(size(obs))
    rand!(groups)

    # create the training and testing observations
    # observation is in test set if random number < holdout_proportion
    train = obs[filter(i->(groups[i]>=holdout_proportion), 1:length(obs))]
    train_observed_features, train_observed_examples = sort_observations(train,m,n)
    test = obs[filter(i->(groups[i]<holdout_proportion), 1:length(obs))]
    test_observed_features, test_observed_examples = sort_observations(test,m,n,check_empty=false)
    return (train_observed_features, train_observed_examples,
                test_observed_features,  test_observed_examples)
end

function flatten_observations(observed_features::ObsArray)
    obs = @compat Array{Tuple{Int,Int}}(0)
    for (i, features_in_example_i) in enumerate(observed_features)
        for j in features_in_example_i
            push!(obs, (i,j))
        end
    end
    return obs
end

function flatten(x, y)
    state = start(x)
    if state==false
        push!(y, x)
    else
        while !done(x, state)
          (item, state) = next(x, state)
          flatten(item, y)
        end
    end
    y
end
flatten{T}(x::Array{T})=flatten(x,Array(T, 0))

function flattenarray(x, y)
    if typeof(x)<:Array
        for xi in x
          flattenarray(xi, y)
        end
    else
        push!(y, x)
    end
    y
end
flattenarray{T}(x::Array{T})=flattenarray(x,Array(T, 0))

function cv_by_iter(glrm::AbstractGLRM, holdout_proportion=.1,
                    params=Params(100,max_iter=1,abs_tol=.01,min_stepsize=.01),
                    ch = ConvergenceHistory("cv_by_iter");
                    verbose=true)
    # obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
    obs = flatten_observations(glrm.observed_features)

    train_observed_features, train_observed_examples, test_observed_features, test_observed_examples =
        get_train_and_test(obs, size(glrm.A)..., holdout_proportion)

    # form glrm on training dataset
    train_glrm = copy_estimate(glrm)
    train_glrm.observed_examples = train_observed_examples
    train_glrm.observed_features = train_observed_features

    # form glrm on testing dataset
    test_glrm = copy_estimate(glrm)
    test_glrm.observed_examples = test_observed_examples
    test_glrm.observed_features = test_observed_features

    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_glrm.observed_features))

    niters = params.max_iter
    params.max_iter = 1
    train_error = @compat Array{Float64}(niters)
    test_error = @compat Array{Float64}(niters)
    if verbose
        @printf("%12s%12s%12s\n", "train error", "test error", "time")
        t0 = time()
    end
    for iter=1:niters
        # evaluate train and test error
        fit!(train_glrm, params, ch=ch, verbose=false)
        train_error[iter] = ch.objective[end] # objective(train_glrm, parameter_estimate(train_glrm)..., include_regularization=false)/ntrain
        test_error[iter] = objective(test_glrm, parameter_estimate(train_glrm)..., include_regularization=false)/ntest
        if verbose
            @printf("%12.4e%12.4e%12.4e\n", train_error[iter], test_error[iter], time() - t0)
        end
    end
    return train_error, test_error
end

function regularization_path(glrm::AbstractGLRM; params=Params(), reg_params=logspace(2,-2,5),
                                         holdout_proportion=.1, verbose=true,
                                         ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
    if verbose println("flattening observations") end
    # obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
    obs = flatten_observations(glrm.observed_features)

    if verbose println("splitting train and test sets") end
    train_observed_features, train_observed_examples, test_observed_features, test_observed_examples =
        get_train_and_test(obs, size(glrm.A)..., holdout_proportion)

    if verbose println("forming train and test GLRMs") end
    # form glrm on training dataset
    train_glrm = copy_estimate(glrm)
    train_glrm.observed_examples = train_observed_examples
    train_glrm.observed_features = train_observed_features

    # form glrm on testing dataset
    test_glrm = copy_estimate(glrm)
    test_glrm.observed_examples = test_observed_examples
    test_glrm.observed_features = test_observed_features

    return regularization_path(train_glrm, test_glrm; params=params, reg_params=reg_params,
                                         verbose=verbose,
                                         ch=ch)
end

# For each value of the regularization parameter,
# compute the training error, ie, average error (sum over (i,j) in train_glrm.obs of L_j(A_ij, x_i y_j))
# and the test error, ie, average error (sum over (i,j) in test_glrm.obs of L_j(A_ij, x_i y_j))
function regularization_path(train_glrm::AbstractGLRM, test_glrm::AbstractGLRM; params=Params(), reg_params=logspace(2,-2,5),
                                         verbose=true,
                                         ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
    train_error = @compat Array{Float64}(length(reg_params))
    test_error = @compat Array{Float64}(length(reg_params))
    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_glrm.observed_features))
    if verbose println("training model on $ntrain samples and testing on $ntest") end
    @show params
    train_time = @compat Array{Float64}(length(reg_params))
    for iparam=1:length(reg_params)
        reg_param = reg_params[iparam]
        # evaluate train and test error
        if verbose println("fitting train GLRM for reg_param $reg_param") end
        scale_regularizer!(train_glrm, reg_param)
        # no need to restart glrm X and Y even if they went to zero at the higher regularization
        # b/c fit! does that automatically
        fit!(train_glrm, params, ch=ch, verbose=verbose)
        train_time[iparam] = ch.times[end]
        if verbose println("computing mean train and test error for reg_param $reg_param:") end
        train_error[iparam] = objective(train_glrm, parameter_estimate(train_glrm)..., include_regularization=false) / ntrain
        if verbose println("\ttrain error: $(train_error[iparam])") end
        test_error[iparam] = objective(test_glrm, parameter_estimate(train_glrm)..., include_regularization=false) / ntest
        if verbose println("\ttest error:  $(test_error[iparam])") end
    end
    return train_error, test_error, train_time, reg_params
end


function precision_at_k(train_glrm::GLRM, test_observed_features; params=Params(), reg_params=logspace(2,-2,5),
                        holdout_proportion=.1, verbose=true,
                        ch::ConvergenceHistory=ConvergenceHistory("reg_path"), kprec=10)
    m,n = size(train_glrm.A)
    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_observed_features))
    train_observed_features = train_glrm.observed_features
    train_error = @compat Array{Float64}(length(reg_params))
    test_error = @compat Array{Float64}(length(reg_params))
    prec_at_k = @compat Array{Float64}(length(reg_params))
    @compat solution = @compat Array{Tuple{Float64,Float64}}(length(reg_params))
    train_time = @compat Array{Float64}(length(reg_params))
    test_glrm = GLRM(train_glrm.A, train_glrm.losses, train_glrm.rx, train_glrm.ry, train_glrm.k,
                     X=copy(train_glrm.X), Y=copy(train_glrm.Y),
                     observed_features = test_observed_features)
    for iparam=1:length(reg_params)
        reg_param = reg_params[iparam]
        # evaluate train error
        if verbose println("fitting train GLRM for reg_param $reg_param") end
        scale!(train_glrm.rx, reg_param)
        scale!(train_glrm.ry, reg_param)
        train_glrm.X, train_glrm.Y = randn(train_glrm.k,m), randn(train_glrm.k,n) # this bypasses the error checking in GLRM(). Risky.
        X, Y, ch = fit!(train_glrm, params, ch=ch, verbose=verbose)
        train_time[iparam] = ch.times[end]
        if verbose println("computing train error and precision at k for reg_param $reg_param:") end
        train_error[iparam] = objective(train_glrm, X, Y, include_regularization=false) / ntrain
        if verbose println("\ttrain error: $(train_error[iparam])") end
        test_error[iparam] = objective(test_glrm, X, Y, include_regularization=false) / ntrain
        if verbose println("\ttest error: $(test_error[iparam])") end
        # precision at k
        XY = X'*Y
        q = sort(XY[:],rev=true)[ntrain] # the ntest+ntrain largest value in the model XY
        true_pos = 0; false_pos = 0
        kfound = 0
        for i=1:m
            if kfound >= kprec
                break
            end
            for j=1:n
                if kfound >= kprec
                    break
                end
                if XY[i,j] >= q
                    # i predict 1 and (i,j) was in my test set and i observed 1
                    if j in test_observed_features[i]
                        true_pos += 1
                        kfound += 1
                    # i predict 1 and i did not observe a 1 (in either my test *or* train set)
                    elseif !(j in train_observed_features[i])
                        false_pos += 1
                        kfound += 1
                    end
                end
            end
        end
        prec_at_k[iparam] = true_pos / (true_pos + false_pos)
        if verbose println("\prec_at_k:  $(prec_at_k[iparam])") end
        solution[iparam] = (sum(X)+sum(Y), sum(abs(X))+sum(abs(Y)))
        if verbose println("\tsum of solution, one norm of solution:  $(solution[iparam])") end
    end
    return train_error, test_error, prec_at_k, train_time, reg_params, solution
end
