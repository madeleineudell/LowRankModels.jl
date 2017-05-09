using LowRankModels
using Base.Test

# Parameters
n = 200
m = 200
r = 5
eta = 0.01
delta = 1e-3

# Generate problem
srand(1)
Um = randn(r, n)
Vm = randn(r, m)
U = Um .+ sqrt(eta) * randn(r, n)
V = Vm .+ sqrt(eta) * randn(r, m)
Y = U' * V + sqrt(delta) * randn(n, m)

# Run algorithm
glrm = GLRM(Y, QuadLoss(), [RemQuadReg(50, Um[:, i]) for i = 1:n], 
    [RemQuadReg(50, Vm[:, j]) for j = 1:m], r)
Uh, Vh, iter_info = fit!(glrm)

mseU, mseV, mseY = mean((U - Uh).^2), mean((V - Vh).^2), mean((Y - Uh' * Vh).^2)
@printf("MSE(U) = %.4g, MSE(V) = %.4g, MSE(Y) = %.4g\n", mseU, mseV, mseY)

# Perform some tests
@test mseU < 1e-3
@test mseV < 1e-3
