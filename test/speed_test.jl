using LowRankModels
import StatsBase: sample

srand(1)

function fit_pca_nucnorm_sparse_nonuniform(m,n,k,s)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = quadreg(.1)
	obsx = [sample(1:int(m/4),int(s/2)), sample(int(m/4)+1:m,s-int(s/2))] 
	obsy = sample(1:n,s)
	obs = [(obsx[i],obsy[i]) for i=1:s]
	glrm = GLRM(A,obs,losses,r,r,k)
	X,Y,ch = fit!(glrm)	
	return A,X,Y,ch
end

n = 1000
nobs = 50000
@time A,X,Y,ch = fit_pca_nucnorm_sparse_nonuniform(n,n,5,nobs);
println(ch.objective)
println(length(ch.objective))
