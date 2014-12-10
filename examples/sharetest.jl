println("loading LowRankModels")
@time @everywhere using LowRankModels

function fit_pca(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = zeroreg()
	glrm = GLRM(A,losses,r,r,k)
	p = Params()
	p.max_iter = 10
	X,Y,ch = fit!(glrm)
	println("Convergence history:",ch.objective)
	println("Time/iter:",ch.times[end]/length(ch.times))
	return A,X,Y,ch
end

@everywhere srand(1)
fit_pca(300,300,10)