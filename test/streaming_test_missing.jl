using LowRankModels
import LowRankModels: keep_rows
import StatsBase: sample, Weights

## generate data
srand(1);
m,n,k = 2000,300,3;
p = .5 # missing probability
kfit = k+1
# variance of measurement
sigmasq = .1

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)

XY = X_real*Y_real; # true values
A = XY + sqrt(sigmasq)*randn(m,n) # noisy values

# missing values
M = sprand(m,n,p)
I,J = findn(M) # observed indices (vectors)
obs = [(I[a],J[a]) for a = 1:length(I)] # observed indices (list of tuples)

# and the model
losses = QuadLoss()
rx, ry = QuadReg(.01), QuadReg(.01);
glrm = GLRM(A,losses,rx,ry,kfit,obs=obs)

T = 1000 # burnin; we'll evaluate results for rows > T
XYsmall = XY[T+1:end,:]

println("SVD initialization")
@time init_svd!(glrm)
X0, Y0 = copy(glrm.X), copy(glrm.Y)
smallglrm = keep_rows(glrm, (T+1):m)
# smallglrm = glrm
# XYsmall = XY
svd_obj = objective(smallglrm, include_regularization=false)
Ahat = impute(smallglrm)
println("After svd init, imputed values differ from true values by $(vecnorm(XYsmall - Ahat)/sqrt(prod(size(Ahat)))) in RMSE\n")

# compile (calls init_svd! and fit! inside fit_streaming!)
println("Small test run (compiling)")
@time fit_streaming!(keep_rows(glrm, 10), StreamingParams(5, Y_update_interval=2))


println("Streaming fit")
@time fit_streaming!(glrm) #, StreamingParams()) # T0, stepsize=.1, Y_update_interval=1))
smallglrm = keep_rows(glrm, (T+1):m)
# smallglrm = glrm
# XYsmall = XY
streaming_obj = objective(smallglrm, include_regularization=false)
Ahat = impute(smallglrm)
println("After streaming fit, imputed values differ from true values by $(vecnorm(XYsmall - Ahat)/sqrt(prod(size(Ahat)))) in RMSE\n")
init_glrm = keep_rows(glrm, T0)
Ainit = impute(init_glrm)
Xs, Ys = copy(glrm.X), copy(glrm.Y)

println("Standard fit")
# glrm.X, glrm.Y = X0, Y0
@time fit!(glrm)
smallglrm = keep_rows(glrm, (T+1):m)
# smallglrm = glrm
# XYsmall = XY
standard_obj = objective(smallglrm, include_regularization=false)
Ahat = impute(smallglrm)
println("After fitting, imputed values differ from true values by $(vecnorm(XYsmall - Ahat)/sqrt(prod(size(Ahat)))) in RMSE\n")
X, Y = copy(glrm.X), copy(glrm.Y)


println("Streaming GLRM performs ", round(svd_obj / streaming_obj, 2), " times better than SVD initialization")
println("Streaming GLRM performs ", round(streaming_obj / standard_obj, 2), " times worse than standard GLRM")

println("Streaming Y differs from standard GLRM Y by ", norm(Ys - Y) / norm(Y), "in RMSE")
println("Streaming X differs from standard GLRM X by ", norm(Xs[:,T:end] - X[:,T:end]) / norm(X[:,T:end]), "in RMSE")
