using LowRankModels
using Distributions
srand(3);

# Define the structure of our heterogenous dataset
# gDataType 		Loss 					n
data_list = [
( gReal(), 			quadratic(), 			1 ),
( gReal(), 			l1(), 					1 ),
( gReal(), 			huber(), 				1 ),
( gPeriodic(1), 	periodic(1), 			1 ),
( gOrdinal(1,10), 	ordinal_hinge(1,10), 	1 ),
( gBool(), 			logistic(), 			1 ),
( gBool(), 			weighted_hinge(),		1 )
]
losses, types = Loss[], gDataType[]
for (t::gDataType, l::Loss, n) in data_list
	losses = [losses, fill(l, n)]
	types = [types, fill(t, n)]
end

# Make a low rank matrix as our data precursor
m, n, true_k = 1000, length(types), 10;
X_real, Y_real = 2*randn(m,k), 2*randn(k,n);
A_real = X_real*Y_real;

# Impute over the low rank-precursor to make our heterogenous dataset
A_no_noise = impute(types, losses, A_real);						# our ideal data
A = impute(types, losses, A_real+1*rand(Cauchy(),size(A_real)));	# our data with noise

p = Params(1, max_iter=400, convergence_tol=0.0001, min_stepsize=0.001)
model_k = 15;
rx, ry = zeroreg(), zeroreg();

# A GLRM using the datatype appropriate losses
hetero = GLRM(A, losses, rx,ry,model_k, scale=true, offset=true);
@time X,Y,ch = fit!(hetero, params=p, verbose=false);
hetero_err = error_metric(types, losses, X'*Y, A_no_noise)

# A GLRM using all quadratic loss w/ no reg (PCA)
base_losses = Array(Loss, size(losses));
fill!(base_losses, quadratic());
base = GLRM(A, base_losses, rx,ry,model_k, scale=true, offset=true);
init_svd!(base)
@time Xb,Yb,chb = fit!(base, params=p, verbose=false);
base_err = error_metric(types, base_losses, Xb'*Yb, A_no_noise)

println()
println("Hetero Error:")
println(ch.objective[end])
println(sum(vec(hetero_err)))
println([sum(hetero_err[:,f]) for f in 1:n])
println("Baseline Error:")
println(chb.objective[end])
println(sum(vec(base_err)))
println([sum(base_err[:,f]) for f in 1:n])

# end

