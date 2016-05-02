using LowRankModels
import StatsBase: sample, WeightVec

println("\n\nspca\n====\n\n")

## generate data
srand(1);
m,n,k = 300,300,5;
file_prefix = "spca_n=300_r=5" #n=$(n)_r=$k"
kfit = k

# coordinates of covariates
X_real = sprandn(m,k,.5)
# directions of observations
Y_real = sprandn(k,n,.5) 

XY = X_real*Y_real;
A = XY

# and the model
losses = QuadLoss()
rx, ry = OneReg(), OneReg();
glrm = GLRM(A,losses,rx,ry,kfit)
glrm_apalm = GLRM(A,losses,rx,ry,kfit)

include("fit_n_plot.jl")