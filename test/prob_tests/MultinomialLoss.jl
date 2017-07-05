using LowRankModels
import StatsBase: sample, Weights

# tests MNL loss

srand(1);
m,n,k = 200,50,2;
kfit = k+1
K = 4; # number of categories
d = n*K;
# matrix to encode
X_real, Y_real = randn(m,k), randn(k,d);
XY = X_real*Y_real;
# subtract the mean so we can compare the truth with the fit;
# the loss function is invariant under shifts
losses = fill(MultinomialLoss(K),n)
yidxs = get_yidxs(losses)
for i=1:m
	for j=1:n
		mef = mean(XY[i,yidxs[j]])
		XY[i,yidxs[j]] = XY[i,yidxs[j]] - mef
	end
end

A = zeros(Int, (m, n))
for i=1:m
	for j=1:n
		wv = Weights(Float64[exp(-XY[i, K*(j-1) + l]) for l in 1:K])
		l = sample(wv)
		A[i,j] = l
	end
end

# and the model
losses = fill(MultinomialLoss(K),n)
rx, ry = QuadReg(), QuadReg();
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,d));

# fit w/o initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y;
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("(Picking randomly, $((K-1)/K*100)\% of entries would be wrong.)\n")

# initialize
init_svd!(glrm)
XYh = glrm.X' * glrm.Y
println("After initialization with the svd, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After initialization with the svd, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("(Picking randomly, $((K-1)/K*100)\% of entries would be wrong.)\n")

# fit w/ initialization
@time X,Y,ch = fit!(glrm);
XYh = X'*Y;
@show ch.objective
println("After fitting, parameters differ from true parameters by $(vecnorm(XY - XYh)/sqrt(prod(size(XY)))) in RMSE")
A_imputed = impute(glrm)
println("After fitting, $(sum(A_imputed .!= A) / prod(size(A))*100)\% of imputed entries are wrong")
println("(Picking randomly, $((K-1)/K*100)\% of entries would be wrong.)\n")
