using LowRankModels

function autoencode_pca(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = zeroreg()
	glrm = GLRM(A,losses,r,r,k)
	X,Y,ch = autoencode!(glrm)
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function autoencode_nnmf(m,n,k)
	# matrix to encode
	A = rand(m,k)*rand(k,n)
	losses = fill(quadratic(),n)
	r = nonnegative()
	glrm = GLRM(A,losses,r,r,k)
	X,Y,ch = autoencode!(glrm)
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function autoencode_pca_nucnorm(m,n,k)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = quadreg(.1)
	glrm = GLRM(A,losses,r,r,k)
	X,Y,ch = autoencode!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function autoencode_kmeans(m,n,k)
	# matrix to encode
	Y = randn(k,n)
	A = zeros(m,n)
	for i=1:m
		A[i,:] = Y[mod(i,k)+1,:]
	end
	losses = fill(quadratic(),n)
	ry = zeroreg()
	rx = onesparse() 
	glrm = GLRM(A,losses,rx,ry,k+4)
	X,Y,ch = autoencode!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

function autoencode_pca_nucnorm_sparse(m,n,k,s)
	# matrix to encode
	A = randn(m,k)*randn(k,n)
	losses = fill(quadratic(),n)
	r = quadreg(.1)
	obsx = sample(1:m,s); obsy = sample(1:n,s)
	obs = [(obsx[i],obsy[i]) for i=1:s]
	glrm = GLRM(A,obs,losses,r,r,k)
	X,Y,ch = autoencode!(glrm)	
	println("Convergence history:",ch.objective)
	return A,X,Y,ch
end

if true
	autoencode_pca(100,100,2)
	autoencode_pca_nucnorm(100,100,2)
	autoencode_pca_nucnorm_sparse(500,500,2,10000)
	autoencode_kmeans(50,50,10)
	autoencode_nnmf(50,50,2)
end