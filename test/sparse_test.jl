using Test
using LowRankModels

# Check that sparse algorithm converges to the same solution
# from the same initial conditions for simple pca.
m,n,k = 100,100,3
A = randn(m,k)*randn(k,n)
loss = QuadLoss()
r = ZeroReg()

glrm_1 = GLRM(A,loss,r,r,k) # solve with prox algorithm
glrm_2 = deepcopy(glrm_1)   # solve with sparse prox algorithm

X1,Y1,ch1 = fit!(glrm_1,ProxGradParams())
X2,Y2,ch2 = fit!(glrm_2,SparseProxGradParams())

@test_approx_eq X1 X2
@test_approx_eq Y1 Y2

# Check that the sparsity pattern in the data is correctly identified.
A = sprand(m,n,0.5)
I,J = findn(A)

glrm = GLRM(A,loss,r,r,k) # create glrm from sparse matrix

@test length(glrm.observed_features) == m
@test length(glrm.observed_examples) == n

for i = 1:m
	for j = 1:n
		if j in glrm.observed_features[i]
			@test A[i,j] != 0.0
		else
			@test A[i,j] == 0.0
		end
	end
end

for j = 1:n
	for i = 1:m
		if i in glrm.observed_examples[j]
			@test A[i,j] != 0.0
		else
			@test A[i,j] == 0.0
		end
	end
end
