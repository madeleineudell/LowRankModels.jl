println("loading LowRankModels")
@time @everywhere using LowRankModels

function fit_pca(m,n,k)
	# matrix to encode
	srand(1)
	A = randn(m,k)*randn(k,n)
	X=randn(k,m)
	Y=randn(k,n)
	losses = fill(QuadLoss(),n)
	r = QuadReg()
	glrm = GLRM(A,losses,r,r,k, X=X, Y=Y)
	glrm = share(glrm)
	p = Params()
	p.max_iter = 10
	X,Y,ch = fit!(glrm)
	println("Convergence history:",ch.objective)
	println("Time/iter:",ch.times[end]/10)
	return A,X,Y,ch
end

@everywhere srand(1)
fit_pca(100,100,50)	