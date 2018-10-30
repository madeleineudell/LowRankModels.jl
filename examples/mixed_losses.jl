using LowRankModels
import StatsBase: sample, Weights

# truncated + logistic loss

##### generate data ######
srand(1);
m,n,k = 300,200,3;
kfit = k+1
# variance of measurement
sigmasq = .1

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)

XY = X_real*Y_real;
XYn = XY + sqrt(sigmasq)*randn(m,n)
A = Array{Number}(size(XY)...)

# first T columns are truncated
T = 150
lb = 0
ub = 10
A[:,1:T] = max.(lb, min.(ub, XYn[:,1:T]))
# the rest are boolean (+1 w/prob logistic(A), -1 otherwise)
for i=1:m
  for j=(T+1):n
    p = 1/(1+exp(-XYn[i,j]))
    p > rand() ? A[i,j] = 1 : A[i,j] = -1
  end
end

###### form model ######
trunc_losses = [TruncatedLoss(QuadLoss(), lb, ub) for i=1:T]
logistic_losses = [LogisticLoss() for i=(T+1):n]
losses = [trunc_losses..., logistic_losses...]
rx, ry = QuadReg(.1), QuadReg(.1);
glrm = GLRM(A,losses,rx,ry,kfit)
#scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,n));

# fit w/custom losses
init_svd!(glrm)
@time X,Y,ch = fit!(glrm);
println("After fitting with custom losses, parameters differ from true parameters by $(vecnorm(XY - X'*Y)/sqrt(prod(size(XY)))) in RMSE\n")
Ahat = impute(glrm);
rmse = norm(A - Ahat) / sqrt(prod(size(A)))
println("Imputations with custom losses differ from true matrix values by $rmse in RMSE")

# fit w/quad loss
glrm.losses = fill(QuadLoss(), length(glrm.losses))
init_svd!(glrm)
@time X,Y,ch = fit!(glrm);
println("After fitting with QuadLoss, parameters differ from true parameters by $(vecnorm(XY - X'*Y)/sqrt(prod(size(XY)))) in RMSE\n")
Ahat = impute(glrm);
rmse = norm(A - Ahat) / sqrt(prod(size(A)))
println("Imputations with QuadLoss differ from true matrix values by $rmse in RMSE")
