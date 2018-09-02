using LowRankModels
using Test

TOL = 1e-5

# QuadConstraint
r = QuadConstraint(7)
@test evaluate(r, ones(7)) == 0
@test evaluate(r, ones(100)) == Inf
@test_approx_eq prox(r, ones(100), 1) ones(100)/sqrt(100)*7

r = KSparseConstraint(3)
u = [-1,2,-3,4,-5.]
@test evaluate(r, u) == Inf
@test prox!(r, u, 1) == [0,0.,-3,4,-5.]
