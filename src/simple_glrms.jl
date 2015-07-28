export pca, qpca, nnmf, rpca, kmeans

# principal components analysis
# minimize ||A - XY||^2
function pca(A::AbstractArray, k::Int)
	loss = quadratic()
	r = zeroreg()
	return GLRM(A,loss,r,r,k)
end

# quadratically regularized principal components analysis
# minimize ||A - XY||^2 + scale*||X||^2 + scale*||Y||^2
function qpca(A::AbstractArray, k::Int; scale=1.0::Float64)
	loss = quadratic()
	r = quadreg(scale)
	return GLRM(A,loss,r,r,k)
end

# nonnegative matrix factorization
# minimize_{X<=0, Y>=0} ||A - XY||^2
function nnmf(A::AbstractArray, k::Int)
	loss = quadratic()
	r = nonnegative()
	return GLRM(A,loss,r,r,k)
end

# k-means
# minimize_{columns of X are unit vectors} ||A - XY||^2
function kmeans(A::AbstractArray, k::Int)
	loss = quadratic()
	ry = zeroreg()
	rx = unitonesparse() 
	return GLRM(A,loss,rx,ry,k)
end

# robust PCA
# minimize huber(A - XY) + scale*||X||^2 + scale*||Y||^2
function rpca(A::AbstractArray, k::Int; scale=1.0::Float64)
	loss = huber()
	r = quadreg(scale)
	return GLRM(A,loss,r,r,k)
end