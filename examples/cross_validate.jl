using LowRankModels, DataFrames

m,n,k = 100,100,3
A = randn(m,k)*randn(k,n)
losses = fill(quadratic(),n)
r = quadreg(.1)
glrm = GLRM(A,losses,r,r,k)
train_error, test_error, train_glrms, test_glrms = cross_validate(glrm,5,Params(1,100,0,.001))

df = DataFrame(train_error = train_error, test_error = test_error)
println(df)