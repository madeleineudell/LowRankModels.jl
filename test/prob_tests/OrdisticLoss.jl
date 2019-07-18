using LowRankModels,  Random
import StatsBase: sample, Weights
import LinearAlgebra: norm 

# test ordistic loss

## generate data
Random.seed!(1);
m,n,k = 200,50,3;
kfit = k+1
d = 7; # number of levels
D = n*d;
# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)
# centers of measurement
T_real = sqrt(k)*randn(d,n) # notice x^T y has variance k, so this scales the thresholds in the same way
for j=1:n
	T_real[:,j] = sort(T_real[:,j])
end
# variance of measurement
sigmasq = 1

XY = X_real*Y_real;
A = zeros(Int, (m, n))
for i=1:m
	for j=1:n
		wv = Weights(Float64[exp(-(XY[i,j] - T_real[l,j])^2/sigmasq) for l in 1:d])
		l = sample(wv)
		A[i,j] = l
	end
end

# and the model
losses = fill(OrdisticLoss(d),n)
rx, ry = lastentry1(QuadReg(.01)), lastentry_unpenalized(QuadReg(.01));
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,D));

# fit w/o initialization
p = Params(1, max_iter=10, abs_tol=0.0000001, min_stepsize=0.000001)
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(norm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs,(A_imputed - A)) / prod(size(A))*100)% on average")
println("(Picking randomly, $((d-1)/d*100)% of entries would be wrong.)\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y[:,1:d:D]
println("After initialization with the svd, parameters differ from true parameters by $(norm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After initialization with the svd, $(sum(A_imputed .!= A) / prod(size(A))*100)% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs,(A_imputed - A)) / prod(size(A))*100)% on average")
println("(Picking randomly, $((d-1)/d*100)% of entries would be wrong.)\n")

# fit w/ initialization
p = Params(1, max_iter=10, abs_tol=0.0000001, min_stepsize=0.000001)
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(norm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs,(A_imputed - A)) / prod(size(A))*100)% on average")
println("(Picking randomly, $((d-1)/d*100)% of entries would be wrong.)\n")
