using LowRankModels
import StatsBase: sample, WeightVec

# test quadratic loss

## generate data
srand(1);
m,n,k = 300,300,3;
kfit = k+1
# variance of measurement
sigmasq = .1

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n) 

XY = X_real*Y_real;
A = XY + sqrt(sigmasq)*randn(m,n)

# and the model
losses = QuadLoss()
rx, ry = QuadReg(.1), QuadReg(.1);
glrm = GLRM(A,losses,rx,ry,kfit)
#scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,n));

# fit w/o initialization
p = Params(1, max_iter=10, convergence_tol=0.0000001, min_stepsize=0.000001) 
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y
println("After initialization with the svd, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE\n")

# fit w/ initialization
p = Params(1, max_iter=10, convergence_tol=0.0000001, min_stepsize=0.000001) 
@time X,Y,ch = fit!(glrm, params=p);
XYh = X'*Y;
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")