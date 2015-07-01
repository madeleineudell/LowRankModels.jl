using LowRankModels

# test losses in losses.jl
srand(1);

losses = [
quadratic(),
quadratic(10),
l1(),
l1(5.2),  
huber(), 
huber(4),
huber(3.1, crossover=3.2),
periodic(2*pi), 
periodic(2*pi, 4),
#poisson(20),
#poisson(22,4.1),
ordinal_hinge(1,10), 
ordinal_hinge(2,7,5),
logistic(),
logistic(0.2),
weighted_hinge(),
weighted_hinge(11),
weighted_hinge(1.5, case_weight_ratio=4.3)
] #tests what should be successful constructions

# TODO: do some bad constructions and test that they fail with catches
bad_losses = [
:(quadratic(1,BoolDomain())),
:(quadratic(10,RealDomain)),
:(huber(3.1, 3.2)),
:(periodic(scale=2*pi)), 
:(periodic(2*pi, scale=4)),
:(periodic())
]
for expression in bad_losses
	try 
		eval(expression);
		println("test FAILED for $expression")
	catch
		println("test PASSED for $expression (failed to construct)")
	end
end

m,n,k = 1000, length(losses), 5;
X_real, Y_real = 2*randn(m,k), 2*randn(k,n);
A_real = X_real*Y_real;

# tests default imputations and implicit domains
# we can visually inspect the differences between A and A_real to make sure imputation is right
A = impute(losses, A_real) 

# tests all the M-estimators with scale=true
glrm = GLRM(A, losses, zeroreg(), zeroreg(), 5, scale=true, offset=true);

# tests eval and grad
@time X,Y,ch = fit!(glrm);

