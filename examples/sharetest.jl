println("loading LowRankModels")
@time @everywhere using LowRankModels

function fit_pca(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = zeroreg()
	glrm = GLRM(A,losses,r,r,k)
	X,Y,ch = fit!(glrm)
	println("Convergence history:",ch.objective)
	println("Time/iter:",ch.times[end]/length(ch.times))
	return A,X,Y,ch
end

@everywhere srand(1)
fit_pca(100,100,10)