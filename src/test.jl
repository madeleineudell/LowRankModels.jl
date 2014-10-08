include("LowRankModels.jl")

function fit_pca_nucnorm_sparse(m,n,r,k,s)
	# matrix to encode
	A = randn(m,r)*randn(r,n)
	losses = fill(quadratic(),n)
	mu = .1
	r = quadreg(mu)
	obsx = sample(1:m,s); obsy = sample(1:n,s)
	obs = [(obsx[i],obsy[i]) for i=1:s]
	glrm = GLRM(A,obs,losses,r,r,k)

	initPoint = zeros(A)
	for i in 1:s
		initPoint[obsx[i],obsy[i]] = A[obsx[i],obsy[i]]
	end
	initX, initD, initY = svd(initPoint)
	glrm.X = (initX * diagm(map(sqrt, initD)))[:, 1:k]
	glrm.Y = transpose((initY * diagm(map(sqrt, initD)))[:, 1:k])

	X,Y,ch = fit!(glrm, Params(1,100,.001), ConvergenceHistory("glrm"))
	if(max(m,n)>=1000 || s / m / n >0.38)
		muf = 1e-4
	else
		muf = 1e-8
	end
	while mu >= muf
		mu = mu * 0.25
		r = quadreg(mu)
		glrm.rx = r
		glrm.ry = r
		X,Y,ch = fit!(glrm, Params(1,100,.0001), ch, false)
	end
	println("Convergence history:",ch.objective[end])
	return A,X,Y,obs, ch
end

function testFitPcaNucnormSparse(m,n,r,k,s,N)
	errRate = Float64[]
	timer = Float64[]
	obj = Float64[]
	for i in 1:N
		tic()
		A, X, Y, obs, ch = fit_pca_nucnorm_sparse(m,n,r,k,s)
		push!(timer,toc())
		push!(errRate, sqrt(sum(map(x->x^2,A - X * Y))) / sqrt(sum(x -> x ^2, A)))
		push!(obj, ch.objective[end])
	end
	return errRate, timer, obj
end


srand(1234)
#test1 = testFitPcaNucnormSparse(40, 40, 3, 3, 800, 5)
# test1bis = testFitPcaNucnormSparse(40, 40, 3, 2, 800, 5)
#test2 = testFitPcaNucnormSparse(100, 100, 10, 10, 5666, 5)
#test3 = testFitPcaNucnormSparse(100, 100, 15, 15, 3000, 5)
test4 = testFitPcaNucnormSparse(200, 200, 10, 10, 15665, 5)
test5 = testFitPcaNucnormSparse(500, 500, 10, 10, 49471, 5)
test6 = testFitPcaNucnormSparse(1000, 1000, 50, 50, 389852, 5)
test7 = testFitPcaNucnormSparse(5000, 5000, 50, 50, 2486747, 5)