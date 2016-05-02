using LowRankModels
import StatsBase: sample, WeightVec

println("\n\nnnmf\n====\n\n")

## generate data
srand(1);
m,n,k = 300,300,5;
file_prefix = "nnmf_n=300_r=3" #n=$(n)_r=$k"
kfit = k

# coordinates of covariates
X_real = rand(m,k)
# directions of observations
Y_real = rand(k,n) 

XY = X_real*Y_real;
A = XY

# and the model
losses = QuadLoss()
rx, ry = NonNegConstraint(), NonNegConstraint();
glrm = GLRM(A,losses,rx,ry,kfit)
glrm_apalm = GLRM(A,losses,rx,ry,kfit)

include("fit_n_plot.jl")