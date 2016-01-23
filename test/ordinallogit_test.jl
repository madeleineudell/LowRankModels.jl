using LowRankModels
import StatsBase: sample, WeightVec

# test ordistic loss

## generate data
srand(1);
m,n,k = 200,100,3;
kfit = k+1
nlevels = 5; # number of levels
d = nlevels-1 # embedding dimension
D = n*d;
# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n) 
# measurement thresholds
T_real = k*randn(d,n) # notice x^T y has variance k; so making this bigger makes the problem easier
for j=1:n # this scheme doesn't work to ensure uniform sampling
	T_real[:,j] = sort(T_real[:,j])
end

signedsums = Array(Float64, d, nlevels)
for i=1:d
    for j=1:nlevels
        signedsums[i,j] = i<j ? 1 : -1
    end
end

XY = X_real*Y_real;
A = zeros(Int, (m, n))
for i=1:m
	for j=1:n
		u = XY[i,j] + T_real[:,j]
		diffs = u'*signedsums
		wv = WeightVec(Float64[exp(-diffs[l]) for l in 1:nlevels])
		l = sample(wv)
		A[i,j] = l
	end
end

# and the model
losses = fill(MultinomialOrdinalLoss(nlevels),n)
rx, ry = lastentry1(QuadReg(.1)), OrdinalReg(QuadReg(.1)) #lastentry_unpenalized(QuadReg(10));
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,D));

# fit w/o initialization
p = Params(1, max_iter=10, convergence_tol=0.0000001, min_stepsize=0.000001) 
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y[:,1:d:D]
println("After initialization with the svd, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After initialization with the svd, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After initialization with the svd, imputed entries are off by $(sum(abs(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")

# fit w/ initialization
p = Params(1, max_iter=10, convergence_tol=0.0000001, min_stepsize=0.000001) 
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")
