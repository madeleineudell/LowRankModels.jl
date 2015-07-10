using LowRankModels
srand(1);

test_losses = Loss[
quadratic(), 	
l1(), 			
huber(), 		
periodic(1), 	
ordinal_hinge(1,10),
logistic(), 		
weighted_hinge()
]

for test_iteration = 1:10
	# Create the configuration for the model (random losses)
	config = int(abs(round(2*rand(length(test_losses)))));
	#config = [0 0 1 0 10 0 100]
	losses, doms = Array(Loss,1), Array(Domain,1);
	for (n,l) in zip(config, test_losses)
		for i=1:n
			push!(losses, l);
			push!(doms, l.domain);
		end
	end
	losses, doms = losses[2:end], doms[2:end]; # this is because the initialization leaves us with an #undef
	my_error_metric(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}) = error_metric(glrm, X, Y, doms, standardize=false) # embed the domains into the error function.
	
	# Make a low rank matrix as our data precursor
	m, n, true_k = 100, length(doms), int(round(length(losses)/2)); 
	X_real, Y_real = randn(true_k,m), randn(true_k,n);
	A_real = X_real'*Y_real;

	# Impute over the low rank-precursor to make our heterogenous dataset
	A = impute(doms, losses, A_real);				# our data with noise

	# Create a glrm using these losses and data
	p = Params(1e-2, max_iter=100, convergence_tol=0.00000001, min_stepsize=1e-15)
	model_k = true_k;
	rx, ry = zeroreg(), zeroreg();
	model = GLRM(A, losses, rx,ry,model_k, scale=false, offset=false);

	# Test that our imputation is consistent
	if my_error_metric(model, X_real, Y_real) != 0
		error("Imputation failed.")
	end

	real_obj = objective(model, X_real, Y_real, include_regularization=false);
	println()
	println("Model:\nk = $(model.k)\nlosses = $(model.losses[1])\nrx,ry = $(model.rx), $(model.ry[1])")
	X_fit,Y_fit,ch = fit!(model, params=p, verbose=false);
	println("Starting objective: $(ch.objective[1])\t Ending objective: $(ch.objective[end])")

	train_err, test_err, trainers, testers = cross_validate(model, error_fn=my_error_metric, verbose=false);
	println("train err: $train_err")
	println("Test err: $test_err")
end
