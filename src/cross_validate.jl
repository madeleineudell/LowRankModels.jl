function cross_validate(glrm::GLRM, nfolds=5, params=Params())
	folds = getfolds(glrm.obs, nfolds)
	train_glrms = Array(GLRM, nfolds)
	test_glrms = Array(GLRM, nfolds)
	train_error = Array(Float64, nfolds)
	test_error = Array(Float64, nfolds)
	for i=1:nfolds
		train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = folds[i]
		# form and fit glrm to training dataset 
		train_glrms[i] = GLRM(glrm.A, train_observed_features, train_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k)
		# form glrm on test dataset, initialize with low rank factors fit to train dataset
		test_glrms[i] = GLRM(glrm.A, test_observed_features, test_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k)
		# evaluate train and test error
		X, Y, ch = fit!(train_glrms[i], params)
		train_error[ifold] = objective(train_glrms[i], X, Y, include_regularization=false)
		test_error[ifold] = objective(test_glrms[i], X, Y, include_regularization=false)
	end
	return train_error, test_error, train_glrms, test_glrms
end

function getfolds(obs, nfolds)
	observed_features, observed_examples = sort_observations(obs,m,n)
	
	# partitions elements of obs into r groups
	groups = Array(Int64, size(obs))
	rand!(1:nfolds, groups)

	# create the training and testing observations for each fold
	folds = Array{Any, nfolds}
	for ifold=1:nfolds
		train = [obs[i] for i=1:length(obs) if groups[i]!=ifold]
		train_observed_features, train_observed_examples = sort_observations(train,m,n)
		test = [obs[i] for i=1:length(obs) if groups[i]==ifold]
		test_observed_features, test_observed_examples = sort_observations(obs,m,n,check_empty=false)
		folds[i] = (train_observed_features, train_observed_examples,
					test_observed_features,  test_observed_examples)
	end
	return folds
end

function cv_by_iter(glrm::GLRM, nfolds=5, params=Params(1,1,.01,.01), niters=100)
	folds = getfolds(glrm.obs, nfolds)
	train_glrms = Array(GLRM, nfolds)
	test_glrms = Array(GLRM, nfolds)
	train_error = Array(Float64, (nfolds, niters))
	test_error = Array(Float64, (nfolds, niters))
	for i=1:nfolds
		train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = folds[i]
		# form and fit glrm to training dataset 
		train_glrms[i] = GLRM(glrm.A, train_observed_features, train_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k)
		# form glrm on test dataset, initialize with low rank factors fit to train dataset
		test_glrms[i] = GLRM(glrm.A, test_observed_features, test_observed_examples, 
			                  glrm.losses, glrm.rx, glrm.ry, glrm.k)
		for iter=1:niters
			# evaluate train and test error
			X, Y, ch = fit!(train_glrms[i], params, verbose=false)
			train_error[ifold,iter] = objective(train_glrms[i], X, Y, include_regularization=false)
			test_error[ifold,iter] = objective(test_glrms[i], X, Y, include_regularization=false)
		end
	end
	return train_error, test_error, train_glrms, test_glrms
end