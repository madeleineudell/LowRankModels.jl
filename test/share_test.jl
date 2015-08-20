@everywhere using LowRankModels
using Base.Test

function fit_pca(m,n,k)
	# matrix to encode
	srand(1)
	# generate a matrix with rank k
	A = randn(m,k)*randn(k,n)
	# fit a PCA model with rank k
	glrm = pca(A, k)
	glrm = share(glrm)
	p = Params()
	# just do 10 iterations
	p.max_iter = 10
	X,Y,ch = fit!(glrm)
	return A,X,Y,ch
end

@everywhere srand(1)
A,X,Y,ch = fit_pca(100,100,50)	

# make sure objective went down
@test ch.objective[end] < ch.objective[1]