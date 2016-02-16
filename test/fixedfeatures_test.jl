using LowRankModels

m,n,k = 10,20,3
Y = rand(3,20)
A = rand(10,20)

ry = Regularizer[fixed_latent_features(Y[:,i]) for i=1:n]
glrm = GLRM(A, QuadLoss(), SimplexConstraint(), ry, k)
X, Yp, ch = fit!(glrm)

@assert(Yp == Y)