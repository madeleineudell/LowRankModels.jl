using LowRankModels

m,n,k = 10,20,3
Y = rand(3,20)
A = rand(10,20)

ry = Regularizer[fixed_latent_features(Y[:,i]) for i=1:n]
glrm = GLRM(A, QuadLoss(), SimplexConstraint(), ry, k+1)
X, Yp, ch = fit!(glrm)

@assert(Yp[1:k,end] == Y)

m,n,k = 10,20,3
Y = rand(3,20)
A = rand(10,20)

ry = Regularizer[fixed_last_latent_features(Y[:,i]) for i=1:n]
glrm = GLRM(A, QuadLoss(), SimplexConstraint(), ry, k+1)
X, Yp, ch = fit!(glrm)

@assert(Yp[2:end,:] == Y)