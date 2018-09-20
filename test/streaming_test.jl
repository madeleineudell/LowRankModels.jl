using LowRankModels
import LowRankModels: keep_rows
import StatsBase: sample, Weights

## generate data
srand(1);
m,n,k = 2000,100,3;
kfit = k
# variance of measurement
sigmasq = 0

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)

XY = X_real*Y_real;
A = XY + sqrt(sigmasq)*randn(m,n)

# and the model
losses = QuadLoss()
rx, ry = QuadReg(.01), QuadReg(.01);
glrm = GLRM(A,losses,rx,ry,kfit);

T = 1000

println("SVD initialization")
@time init_svd!(glrm)
X0, Y0 = copy(glrm.X), copy(glrm.Y)
svd_obj = objective(keep_rows(glrm, (T+1):m), include_regularization=false)

println("Streaming fit")
@time fit_streaming!(glrm, StreamingParams(T, Y_update_interval=100))
streaming_obj = objective(keep_rows(glrm, (T+1):m), include_regularization=false)

println("Standard fit")
glrm.X, glrm.Y = X0, Y0
@time fit!(glrm)
standard_obj = objective(keep_rows(glrm, (T+1):m), include_regularization=false)

println("Streaming GLRM performs ", round(streaming_obj / svd_obj, 2), " times worse than SVD initialization")
println("Streaming GLRM performs ", round(streaming_obj / standard_obj, 2), " times worse than standard GLRM")

smallglrm = keep_rows(glrm, (T+1):m);
sAhat = impute(smallglrm);
sA = smallglrm.A;
norm(sAhat - sA) / norm(sA)
