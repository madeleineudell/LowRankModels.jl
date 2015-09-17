using LowRankModels
using Distributions

# This is just to create a low-rank representation of some count data, I don't care how accurate it is.
m,n = 100, 50;
k = 2;
A = rand(Poisson(),(m,n));
losses = convert(Array{Loss,1}, fill(poisson(10),n));
rx, ry = ZeroReg(), ZeroReg();
g_pre = GLRM(A, losses, rx, ry, k, scale=false, offset=false);
X_real, Y_real, ch = fit!(g_pre, params=Params(0.00001,min_stepsize=0.00000000001,max_iter=10000));

# Now we do the actual model using the perfect data and try to recapture it.
A_real = impute(losses, X_real'*Y_real);

g = GLRM(A_real, losses, rx, ry, k, scale=true, offset=true);
X, Y, ch = fit!(g, params=Params(0.00001,min_stepsize=0.00000000001,max_iter=10000));
U = X'*Y;
A_imputed = impute(losses, X'*Y);

error_metric(g)
errors(Domain[l.domain for l in losses], losses, U, A_real);