using LowRankModels
# tests basic functionality of glrm.jl
Random.seed!(1);
m,n,k,s = 100,100,5,100*100;
# matrix to encode
X_real, Y_real = randn(m,k), randn(k,n);
A = X_real*Y_real;
losses = fill(QuadLoss(),n)
rx, ry = ZeroReg(), ZeroReg();
glrm = GLRM(A,losses,rx,ry,5, scale=false, offset=false, X=randn(k,m), Y=randn(k,n));

p = Params(1, max_iter=200, abs_tol=0.0000001, min_stepsize=0.001)
@time X,Y,ch = fit!(glrm, params=p);
Ah = X'*Y;
p.abs_tol > abs(norm(A-Ah)^2 - ch.objective[end])

function validate_folds(trf,tre,tsf,tse)
	for i=1:length(trf)
		if length(intersect(Set(trf[i]), Set(tsf[i]))) > 0
			println("Error on example $i: train and test sets overlap")
		end
	end
	for i=1:length(tre)
		if length(intersect(Set(tre[i]), Set(tse[i]))) > 0
			println("Error on feature $i: train and test sets overlap")
		end
	end
	true
end

obs = LowRankModels.flatten_observations(glrm.observed_features)
folds = LowRankModels.getfolds(obs, 5, size(glrm.A)..., do_check = false)
for i in 1:length(folds)
	@assert validate_folds(folds[i]...)
end
