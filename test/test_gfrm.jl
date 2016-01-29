using LowRankModels, Compat

srand(1)
# parameters
m, n, k = 20, 20, 3
lambda = 10

# generate data
X0 = randn(m,k)
Y0 = randn(k,n)
A = X0*Y0 + .1*randn(m,n)
P = sparse(float(randn(m,n) .>= .5)) # observed entries

I,J = findn(P) # observed indices (vectors)
@compat obs = Tuple{Int,Int}[(I[a],J[a]) for a = 1:length(I)]
observed_features, observed_examples = sort_observations(obs,size(P)...)

losses = fill(QuadLoss(1), n)
reg = MaxNormReg(lambda)
gfrm = GFRM(A, losses, reg, k, observed_features, observed_examples, 
	zeros(m,n), zeros(m+n,m+n))

fit!(gfrm, PrismaParams(PrismaStepsize(Inf), 100, 1))

# compare with SDP solver
using Convex

W = Variable(m+n, m+n)
obj = sumsquares(P.*(W[1:m, m+1:end]-A)) + lambda*maximum(diag(W))
p = minimize(obj, 
	W in :SDP)
solve!(p)

println("mean square error is ", vecnorm(W.value - gfrm.W) / vecnorm(W.value))
W.value = gfrm.W
println("objective of Convex problem evaluated at prisma solution is $(Convex.evaluate(obj))")
