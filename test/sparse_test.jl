using Base.Test
using LowRankModels

# simple synthetic dataset for PCA
m,n,k = 100,100,3
A = randn(m,k)*randn(k,n)
loss = quadratic()
r = zeroreg()


# Check that sparse algorithm converges to the same solution
# from the same initial conditions.
glrm_1 = GLRM(A,loss,r,r,k)
glrm_2 = deepcopy(glrm_1)

X1,Y1,ch1 = fit!(glrm_1,ProxGradParams())
X2,Y2,ch2 = fit!(glrm_2,SparseProxGradParams())

@test_approx_eq X1 X2
@test_approx_eq Y1 Y2
