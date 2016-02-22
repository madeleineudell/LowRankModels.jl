using LowRankModels, Compat

srand(1)
# parameters
m, n, k = 30, 30, 3
lambda = 1

# generate data
X0 = randn(m,k)
Y0 = randn(k,n)
A = X0*Y0 + .1*randn(m,n)
P = sparse(float(randn(m,n) .>= .5)) # observed entries

I,J = findn(P) # observed indices (vectors)
@compat obs = Tuple{Int,Int}[(I[a],J[a]) for a = 1:length(I)]
observed_features, observed_examples = sort_observations(obs,size(P)...)

losses = fill(QuadLoss(1), n)
reg = TraceNormReg(lambda)
gfrm = GFRM(A, losses, reg, k, observed_features, observed_examples, 
	zeros(m,n), zeros(m+n,m+n))

fit!(gfrm, PrismaParams(PrismaStepsize(Inf), 200, 1))

# compare with SDP solver
using Convex

U = Variable(m,n)
W = Variable(m+n, m+n)

obj = sumsquares(P.*(U-A)) + lambda*nuclearnorm(U)
p = minimize(obj)
solve!(p)

@show Convex.evaluate(lambda*nuclearnorm(U))
println("mean square error is ", vecnorm(U.value - gfrm.U) / vecnorm(U.value))
U.value = gfrm.U
println("objective of Convex problem evaluated at prisma solution is $(Convex.evaluate(obj))
")