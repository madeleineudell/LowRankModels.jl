using LowRankModels
using Distributions

p = Params(0.00001,min_stepsize=0.00000000001,max_iter=5000)

# This is just to create a low-rank representation of some count data, I don't care how accurate it is.
m,n = 100, 50;
k = 2;
A = rand(Poisson(2), m, n); # Data is Poisson with mean 2
losses = convert(Array{Loss,1}, fill(PoissonLoss(10),n));
rx, ry = QuadReg(), QuadReg();
g_pre = GLRM(A, losses, rx, ry, k, scale=false, offset=false);

# let's check a different syntax works, too
g_pre = GLRM(A, PoissonLoss(), rx, ry, k, scale=false, offset=false);
X_real, Y_real, ch = fit!(g_pre, params=p);

# Now we do the actual model using the perfect data and try to recapture it.
A_real = impute(losses, X_real'*Y_real);

g = GLRM(A_real, losses, rx, ry, k, scale=true, offset=true);
X, Y, ch = fit!(g, params=p);
U = X'*Y;
A_imputed = impute(losses, X'*Y);

@show error_metric(g)
errors(Domain[l.domain for l in losses], losses, U, A_real);
