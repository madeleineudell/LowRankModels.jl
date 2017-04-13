using LowRankModels
srand(1);

test_losses = Loss[
QuadLoss(), 	
L1Loss(), 			
HuberLoss(), 		
PeriodicLoss(1), 	
OrdinalHingeLoss(1,10),
LogisticLoss(), 		
WeightedHingeLoss()
]

for test_iteration = 1:500
	# Create the configuration for the model (random losses)
	config = round.(Int, abs(round(4*rand(length(test_losses)))));
	#config = [0 0 1 0 10 0 100]
	losses, doms = Array(Loss,1), Array(Domain,1);
	for (n,l) in zip(config, test_losses)
		for i=1:n
			push!(losses, l);
			push!(doms, l.domain);
		end
	end
	losses, doms = losses[2:end], doms[2:end]; # this is because the initialization leaves us with an #undef
	my_error_metric(glrm::GLRM, X::Array{Float64,2}, Y::Array{Float64,2}) = error_metric(glrm, X, Y, doms, standardize=true) # embed the domains into the error function.
	
	# Make a low rank matrix as our data precursor
	m, n, true_k = 100, length(doms), round.(Int, round(length(losses)/2))+1; 

	X_real, Y_real = randn(true_k,m), randn(true_k,n);
	A_real = X_real'*Y_real;

	# Impute over the low rank-precursor to make our heterogenous dataset
	A = impute(doms, losses, A_real);				# our data with noise

	# Create a glrm using these losses and data
	p = Params(1e-2, max_iter=1000, abs_tol=0.00000001, min_stepsize=1e-15)
	rx, ry = ZeroReg(), ZeroReg();

	k_range = [int(round(true_k/2)), true_k]
	train_err_at_k = Dict()
	println("\n########################################################")
	println("Model:\nlosses = $(losses)\nrx,ry = $(rx), $(ry)")
	for k in k_range
		model = GLRM(A, losses, rx,ry,k, scale=false, offset=false);
		println("\nk = $(k)")
		# Test that our imputation is consistent
		if my_error_metric(model, X_real, Y_real) != 0
			error("Imputation failed.")
		end

		real_obj = objective(model, X_real, Y_real, include_regularization=false);
		X_fit,Y_fit,ch = fit!(model, params=p, verbose=false);
		println("Starting objective: $(ch.objective[1])\t Ending objective: $(ch.objective[end])")

		train_err, test_err, trainers, testers = cross_validate(model, nfolds=3, error_fn=my_error_metric, verbose=false);
		train_err_at_k[k] = my_error_metric(model, model.X, model.Y)
		println("train err: $train_err")
		println("Test err: $test_err")
	end
	train_err_at_k = [train_err_at_k[k] for k in k_range]
	if !all([de<0 for de in diff(train_err_at_k)]) #check that errors monotonically decrease as k increases
		warn("==================================================================================")
		warn("ERRORS WENT UP FOR THIS CONFIGURATION")
		warn("Model:\nlosses = $(losses)\nrx,ry = $(rx), $(ry)")
		warn("Ranks: $(string(k_range))")
		warn("Training errors: $(string(train_err_at_k))")
		warn("==================================================================================")
	end
end
