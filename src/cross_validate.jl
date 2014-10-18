export cross_validate, cv_by_iter, regularization_path

# To do:
# 	* implement cv_by_iter
# 	* implement regularization_path
# 	* robustify to cv splits that empty a given row or column (eg check and try again)

function cross_validate(glrm::GLRM, nfolds=5, params=Params(); verbose=true)
	if verbose println("flattening observations") end
	obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
	if verbose println("computing CV folds") end
	folds = getfolds(obs, nfolds, size(glrm.A)...)
	train_glrms = Array(GLRM, nfolds)
	test_glrms = Array(GLRM, nfolds)
	train_error = Array(Float64, nfolds)
	test_error = Array(Float64, nfolds)
	for ifold=1:nfolds
		if verbose println("\nforming train and test GLRM for fold $ifold") end
		train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = folds[ifold]
		# form glrm on training dataset 
		train_glrms[ifold] = GLRM(glrm.A, train_observed_features, train_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))
		# form glrm on testing dataset
		test_glrms[ifold] = GLRM(glrm.A, test_observed_features, test_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))
		# evaluate train and test error
		if verbose println("fitting train GLRM for fold $ifold") end
		X, Y, ch = fit!(train_glrms[ifold], params, verbose=false)
		if verbose println("computing train and test error for fold $ifold:") end
		train_error[ifold] = objective(train_glrms[ifold], X, Y, include_regularization=false)
		if verbose println("\ttrain error: $(train_error[ifold])") end
		test_error[ifold] = objective(test_glrms[ifold], X, Y, include_regularization=false)
		if verbose println("\ttest error:  $(test_error[ifold])") end
	end
	return train_error, test_error, train_glrms, test_glrms
end

function getfolds(obs, nfolds, m, n)	

	# partition elements of obs into nfolds groups
	groups = Array(Int64, size(obs))
	rand!(1:nfolds, groups)

	# create the training and testing observations for each fold
	folds = Array(Tuple, nfolds)
	for ifold=1:nfolds
		train = obs[filter(i->groups[i]!=ifold, 1:length(obs))]
		train_observed_features, train_observed_examples = sort_observations(train,m,n)
		test = obs[filter(i->groups[i]==ifold, 1:length(obs))]
		test_observed_features, test_observed_examples = sort_observations(obs,m,n,check_empty=false)
		folds[ifold] = (train_observed_features, train_observed_examples,
					test_observed_features,  test_observed_examples)
	end
	return folds
end

function get_train_and_test(obs, m, n, holdout_proportion=.1)	

	# generate random uniform number for each observation
	groups = Array(Float64, size(obs))
	rand!(groups)

	# create the training and testing observations
	# observation is in test set if random number < holdout_proportion
	train = obs[filter(i->groups[i]>=holdout_proportion, 1:length(obs))]
	train_observed_features, train_observed_examples = sort_observations(train,m,n)
	test = obs[filter(i->groups[i]<holdout_proportion, 1:length(obs))]
	test_observed_features, test_observed_examples = sort_observations(obs,m,n,check_empty=false)
	return (train_observed_features, train_observed_examples,
				test_observed_features,  test_observed_examples)
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

function cv_by_iter(glrm::GLRM, holdout_proportion=.1, params=Params(1,1,.01,.01), niters=30; verbose=true)
	if verbose println("flattening observations") end
	obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
	
	if verbose println("splitting train and test sets") end
	train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = 
		get_train_and_test(obs, size(glrm.A)..., holdout_proportion)
	
	if verbose println("forming train and test GLRMs") end
	# form glrm on training dataset 
	train_glrm = GLRM(glrm.A, train_observed_features, train_observed_examples, 
		                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))
	# form glrm on testing dataset
	test_glrm = GLRM(glrm.A, test_observed_features, test_observed_examples, 
		                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))

	train_error = Array(Float64, niters)
	test_error = Array(Float64, niters)
	for iter=1:niters
		# evaluate train and test error
		if verbose println("fitting train GLRM") end
		X, Y, ch = fit!(train_glrm, params, verbose=false)
		if verbose println("computing train and test error for iter $iter:") end
		train_error[iter] = objective(train_glrm, X, Y, include_regularization=false)
		if verbose println("\ttrain error: $(train_error[iter])") end
		test_error[iter] = objective(test_glrm, X, Y, include_regularization=false)
		if verbose println("\ttest error:  $(test_error[iter])") end
	end
	return train_error, test_error
end

function regularization_path(glrm::GLRM; params=Params(), reg_params=logspace(2,-2,5), 
	                                     holdout_proportion=.1, verbose=true,
	                                     ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
	if verbose println("flattening observations") end
	obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
	
	if verbose println("splitting train and test sets") end
	train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = 
		get_train_and_test(obs, size(glrm.A)..., holdout_proportion)
	
	if verbose println("forming train and test GLRMs") end
	# form glrm on training dataset 
	train_glrm = GLRM(glrm.A, train_observed_features, train_observed_examples, 
		                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))
	# form glrm on testing dataset
	test_glrm = GLRM(glrm.A, test_observed_features, test_observed_examples, 
		                  glrm.losses, glrm.rx, glrm.ry, glrm.k, copy(glrm.X), copy(glrm.Y))

	train_error = Array(Float64, length(reg_params))
	test_error = Array(Float64, length(reg_params))
	train_time = Array(Float64, length(reg_params))
	for iparam=1:length(reg_params)
		reg_param = reg_params[iparam]
		# evaluate train and test error
		if verbose println("fitting train GLRM for reg_param $reg_param") end
		train_glrm.rx.scale, train_glrm.ry.scale = reg_param, reg_param
		X, Y, ch = fit!(train_glrm, params, ch, verbose=verbose)
		train_time[iparam] = ch.times[end]
		if verbose println("computing train and test error for reg_param $reg_param:") end
		train_error[iparam] = objective(train_glrm, X, Y, include_regularization=false)
		if verbose println("\ttrain error: $(train_error[iparam])") end
		test_error[iparam] = objective(test_glrm, X, Y, include_regularization=false)
		if verbose println("\ttest error:  $(test_error[iparam])") end
	end
	return train_error, test_error, train_time, reg_params
end


function regularization_path(train_glrm::GLRM, test_glrm::GLRM; params=Params(), reg_params=logspace(2,-2,5), 
	                                     holdout_proportion=.1, verbose=true,
	                                     ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
	train_error = Array(Float64, length(reg_params))
	test_error = Array(Float64, length(reg_params))
	train_time = Array(Float64, length(reg_params))
	for iparam=1:length(reg_params)
		reg_param = reg_params[iparam]
		# evaluate train and test error
		if verbose println("fitting train GLRM for reg_param $reg_param") end
		train_glrm.rx.scale, train_glrm.ry.scale = reg_param, reg_param
		X, Y, ch = fit!(train_glrm, params, ch, verbose=verbose)
		train_time[iparam] = ch.times[end]
		if verbose println("computing train and test error for reg_param $reg_param:") end
		train_error[iparam] = objective(train_glrm, X, Y, include_regularization=false)
		if verbose println("\ttrain error: $(train_error[iparam])") end
		test_error[iparam] = objective(test_glrm, X, Y, include_regularization=false)
		if verbose println("\ttest error:  $(test_error[iparam])") end
	end
	return train_error, test_error, train_time, reg_params
end