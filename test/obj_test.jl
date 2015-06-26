using LowRankModels

test_losses = Loss[
quadratic(), 	
l1(), 			
huber(), 		
periodic(1), 	
ordinal_hinge(1,10),
logistic(), 		
weighted_hinge()
]

#for test_iteration = 1:5
	# Create the configuration for the model (random losses)
	config = int(abs(round(5*rand(length(test_losses)))));
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
	# fill!(losses, quadratic())
	# doms = Domain[l.domain for l in losses]

	# Make a low rank matrix as our data precursor
	m, n, true_k = 1000, length(doms), int(round(length(losses)/2)); 
	X_real, Y_real = 2*randn(m,true_k), 2*randn(true_k,n);
	A_real = X_real*Y_real;

	# Impute over the low rank-precursor to make our heterogenous dataset
	A = impute(doms, losses, A_real);				# our imputed data

	p = Params(1, max_iter=1000, convergence_tol=0.000001, min_stepsize=0.001);
	rx, ry = zeroreg(), zeroreg();
		
	skip = 5
	k0=skip
	model = GLRM(A, losses, rx, ry, k0, scale=false, offset=false);
	X_fit, Y_fit, ch = fit!(model, params=p, verbose=false);
	Xi = [X_fit; 0.1*randn(skip, m)];
	Yi = [Y_fit; 0.1*randn(skip, n)];
	println("k=$k0")
	println("Obj: $(ch.objective[1]) to $(ch.objective[end])")
	#display([norm(Y_fit[i,:]) for i in 1:k])
	for k = (k0+skip):skip:(true_k*2)
		println()
		println("k=$k")
		model = GLRM(A, losses, rx, ry, k, scale=false, offset=false, X=Xi, Y=Yi);
		naive_model = GLRM(A, losses, rx, ry, k, scale=false, offset=false)
		fix_latent_features!(model, k-skip) # fix the first k-1 rows of Y so we're only finding skip# new ones
		old_ch = ch
		X_fit, Y_fit, ch = fit!(model, params=p, verbose=false);
		X_naive, Y_naive, ch_naive = fit!(naive_model, params=p, verbose=false)
		println("Obj: $(ch.objective[1]) ... $(ch.objective[end]): $(old_ch.objective[end]/ch.objective[end])-fold reduction")
		println("Naive objective: $(ch_naive.objective[end])")
		#display([norm(Y_fit[i,:]) for i in 1:k])
		Xi = [X_fit; 0.1*randn(skip, m)];
		Yi = [Y_fit; 0.1*randn(skip, n)];
	end

#end
# function cosdisty(Y,a,b)
# 	for a=1:k
# 		for b=1:k
# 			dists[a,b] = dot(Y[a,:], Y[b,:]) / (norm(Y[a,:]) * norm(Y[b,:]))
# 		end
# 	end
# 	return dists
# end
# HEATMAP OF DISTS for Y and Y naive PLZ!!!

