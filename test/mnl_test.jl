using LowRankModels
import StatsBase: sample, WeightVec

# tests MNL loss

srand(1);
m,n,k = 1000,300,2;
K = 3; # number of categories
d = n*K;
# matrix to encode
X_real, Y_real = randn(m,k), randn(k,d);
XY = X_real*Y_real;
A = zeros(Int, (m, n))
for i=1:m
	for j=1:n
		wv = WeightVec(Float64[exp(-XY[i, K*(j-1) + l]) for l in 1:K])
		l = sample(wv)
		A[i,j] = l
	end
end

# and the model
losses = fill(MultinomialLoss(K),n)
rx, ry = QuadReg(), QuadReg();
glrm = GLRM(A,losses,rx,ry,k, scale=false, offset=false, X=randn(k,m), Y=randn(k,d));

init_svd!(glrm)
p = Params(1, max_iter=200, convergence_tol=0.0000001, min_stepsize=0.001) 
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y;
@show ch.objective
@show vecnorm(XY - XYh)/prod(size(XY))