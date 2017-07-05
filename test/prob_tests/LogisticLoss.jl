using LowRankModels
import StatsBase: sample, Weights

# test quadratic loss

## generate data
srand(1);
m,n,k = 1000,1000,3;
kfit = k+1
# variance of measurement
sigmasq = .1

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)

XY = X_real*Y_real;
A = zeros(Int, (m, n))
logistic(x) = 1/(1+exp(-x))
for i=1:m
	for j=1:n
		A[i,j] = (logistic(XY[i,j]) >= rand()) ? true : false
	end
end

# and the model
losses = LogisticLoss()
rx, ry = QuadReg(.1), QuadReg(.1);
glrm = GLRM(A,losses,rx,ry,kfit)
#scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,n));

# fit w/o initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs.(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, 50\% of entries would be wrong.)\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y
println("After initialization with the svd, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs.(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, 50\% of entries would be wrong.)\n")

# fit w/ initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs.(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, 50\% of entries would be wrong.)\n")
