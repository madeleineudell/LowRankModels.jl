using LowRankModels
import StatsBase: sample

println("simple glrm examples")

# minimize ||A - XY||^2
function fit_pca(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	loss = quadratic()
	r = zeroreg()
	glrm = GLRM(A,loss,r,r,k)
	X,Y,ch = fit!(glrm)
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

# minimize_{X<=0, Y>=0} ||A - XY||^2
function fit_nnmf(m,n,k)
	# matrix to encode
	A = rand(m,k)*rand(k,n)
	loss = quadratic()
	r = nonnegative()
	glrm = GLRM(A,loss,r,r,k)
	X,Y,ch = fit!(glrm)
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

# minimize ||A - XY||^2 + .1||X||^2 + .1||Y||^2
function fit_pca_nucnorm(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	loss = quadratic()
	r = quadreg(.1)
	glrm = GLRM(A,loss,r,r,k)
	X,Y,ch = fit!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

# minimize_{X<=0} ||A - XY||^2
function fit_kmeans(m,n,k)
	# matrix to encode
	Y = randn(k,n)
	A = zeros(m,n)
	for i=1:m
		A[i,:] = Y[mod(i,k)+1,:]
	end
	loss = quadratic()
	ry = zeroreg()
	rx = unitonesparse() 
	glrm = GLRM(A,loss,rx,ry,k+4)
	X,Y,ch = fit!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function fit_pca_nucnorm_sparse(m,n,k,s)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	loss = quadratic()
	r = quadreg(.1)
	obsx = sample(1:m,s); obsy = sample(1:n,s)
	obs = [(obsx[i],obsy[i]) for i=1:s]
	glrm = GLRM(A,loss,r,r,k, obs=obs)
	X,Y,ch = fit!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function fit_pca_nucnorm_sparse_nonuniform(m,n,k,s)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	loss = quadratic()
	r = quadreg(.1)
	obsx = [sample(1:int(m/4),int(s/2)), sample(int(m/4)+1:m,s-int(s/2))] 
	obsy = sample(1:n,s)
	obs = [(obsx[i],obsy[i]) for i=1:s]
	glrm = GLRM(A,loss,r,r,k, obs=obs)
	X,Y,ch = fit!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function fit_soft_kmeans(m,n,k)
	# PCA with loadings constrained to lie on unit simplex
	# constrain columns of X to lie on unit simplex
	Xreal = rand(k,m)
	Xreal ./= sum(Xreal,1)
	A = Xreal' * randn(k,n)

	loss = quadratic()
	rx = simplex()
	ry = zeroreg()
	glrm = GLRM(A,loss,rx,ry,k)
	X,Y,ch = fit!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

if true
	srand(10)
	fit_pca(100,100,2)
	fit_pca_nucnorm(100,100,2)
	fit_pca_nucnorm_sparse(500,500,2,10000)
	fit_pca_nucnorm_sparse_nonuniform(1000,1000,2,20000)
	fit_kmeans(50,50,10)
	fit_nnmf(50,50,2)
end