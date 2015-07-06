using LowRankModels

m,n,k = 10,20,3
A = rand(m,k)*rand(k,n)
losses = fill(quadratic(),n)
r = nonnegative()
glrm = GLRM(A,losses,r,r,k)
init_nnmf!(glrm)

@test(all(glrm.X .>= 0.0))
@test(all(glrm.Y .>= 0.0))
