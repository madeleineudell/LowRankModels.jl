using LowRankModels
import StatsBase: sample, WeightVec

println("\n\nmultinomial ordinal\n===================\n\n")


## generate data
srand(1);
m,n,k = 300,300,5;
kfit = k
file_prefix = "multinomialordinal_n=300_r=5" #n=$(n)_r=$k"

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
XYplusT = zeros(Float64, (m,D))
A = zeros(Int, (m, n))
for i=1:m
    for j=1:n
        u = XY[i,j] + T_real[:,j]
        XYplusT[i,(j-1)*d+(1:d)] = u 
        diffs = u'*signedsums
        wv = WeightVec(Float64[exp(-diffs[l]) for l in 1:nlevels])
        l = sample(wv)
        A[i,j] = l
    end
end
# loss is insensitive to shifts; regularizer should pick this shift
XYplusT = XYplusT .- mean(XYplusT, 2)

# and the model
losses = fill(MultinomialOrdinalLoss(nlevels),n)
rx, ry = lastentry1(QuadReg(.1)), OrdinalReg(QuadReg(.1)) #lastentry_unpenalized(QuadReg(10));
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,D));
glrm_apalm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,D));

include("fit_n_plot.jl")