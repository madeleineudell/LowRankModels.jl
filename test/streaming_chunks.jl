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

# form two glrms on the first and second half of the data
glrm1 = keep_rows(glrm, 1:1000)
glrm2 = keep_rows(glrm, 1001:2000)

# fit glrm1 by streaming
p1 = StreamingParams()
fit_streaming!(glrm1, p1)

# fit glrm2 by streaming, starting at final parameters of glrm1
glrm2.Y = copy(glrm1.Y)
p2 = StreamingParams(0, stepsize=p1.stepsize) # use zero rows to initialize
fit_streaming!(glrm2, p2)
