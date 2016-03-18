using LowRankModels
using Plotly

test_losses = Loss[
QuadLoss(), 	
L1Loss(), 			
HuberLoss(), 		
PeriodicLoss(1), 	
OrdinalHingeLoss(1,10),
WeightedHingeLoss()
LogisticLoss(), 		
]

#for test_iteration = 1:5
	# Create the configuration for the model (random losses)
	config = round(Int, abs(round(5*rand(length(test_losses)))));
	# config = [1,1,1,1,1,1,10]
	losses, doms = Array(Loss,1), Array(Domain,1);
	for (n,l) in zip(config, test_losses)
		for i=1:n
			push!(losses, l);
			push!(doms, l.domain);
		end
	end
	losses, doms = losses[2:end], doms[2:end]; # this is because the initialization leaves us with an #undef
	# losses = Array(Loss, 20)
	# fill!(losses, QuadLoss())
	# doms = Domain[l.domain for l in losses]

	# Make a low rank matrix as our data precursor
	m, n, true_k = 1000, length(doms), round(Int, round(length(losses)/2)); 
	X_real, Y_real = 2*randn(m,true_k), 2*randn(true_k,n);
	A_real = X_real*Y_real;

	# Impute over the low rank-precursor to make our heterogenous dataset
	A = impute(doms, losses, A_real);				# our imputed data

	p = Params(1, max_iter=1000, abs_tol=0.000001, min_stepsize=0.001);
	rx, ry = ZeroReg(), ZeroReg();
		
	skip = 5
	k0=skip
	model = GLRM(A, losses, rx, ry, k0, scale=false, offset=false);
	X_fit, Y_fit, ch = fit!(model, params=p, verbose=false);


