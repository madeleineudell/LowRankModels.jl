using LowRankModels
# tests basic functionality of glrm.jl
srand(1);
m,n,k,s = 1000,1000,5,1000*1000;
# matrix to encode
X_real, Y_real = randn(m,k), randn(k,n);
A = X_real*Y_real;
losses = fill(QuadLoss(),n)
rx, ry = ZeroReg(), ZeroReg();
glrm = GLRM(A,losses,rx,ry,5, scale=false, offset=false, X=randn(k,m), Y=randn(k,n));

p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001) 
@time X,Y,ch = fit!(glrm, params=p);
Ah = X'*Y;
p.abs_tol > abs(vecnorm(A-Ah)^2 - ch.objective[end])