using LowRankModels
import StatsBase: sample, Weights

# test MNL Ordinal loss

## generate data
srand(1);
m,n,k = 100,100,2;
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

signedsums = @compat Array{Float64}(d, nlevels)
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
		wv = Weights(Float64[exp(-diffs[l]) for l in 1:nlevels])
		l = sample(wv)
		A[i,j] = l
	end
end
# loss is insensitive to shifts; regularizer should pick this shift
XYplusT = XYplusT .- mean(XYplusT, 2)

# and the model
losses = fill(MultinomialOrdinalLoss(nlevels),n)
rx, ry = lastentry1(QuadReg(.01)), MNLOrdinalReg(QuadReg(.01)) #lastentry_unpenalized(QuadReg(10));
Yord = randn(kfit,D)
yidxs = get_yidxs(losses)
for j=1:n
	prox!(ry, view(Yord,:,yidxs[j]), 1)
end
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=Yord);

# fit w/o initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y#[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XYplusT - XYh)/sqrt(prod(size(XYplusT)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs.(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y
println("After initialization with the svd, parameters differ from true parameters by $(vecnorm(XYplusT - XYh)/sqrt(prod(size(XYplusT)))) in RMSE")
A_imputed = impute(glrm)
println("After initialization with the svd, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After initialization with the svd, imputed entries are off by $(sum(abs,(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")

# fit w/ initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y#[:,1:d:D];
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XYplusT - XYh)/sqrt(prod(size(XYplusT)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("After fitting, imputed entries are off by $(sum(abs,(A_imputed - A)) / prod(size(A))*100)\% on average")
println("(Picking randomly, $((nlevels-1)/nlevels*100)\% of entries would be wrong.)\n")

# test scaling
scale!(glrm.ry, 3)
