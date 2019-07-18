using Test
using LowRankModels

## Tests for NNDSVD initialization

# apply init_nnmf! to random dataset
m,n,k = 10,5,3
A = rand(m,k)*rand(k,n)
losses = fill(QuadLoss(),n)
r = NonNegConstraint()
glrm = GLRM(A,losses,r,r,k)
init_nndsvd!(glrm)

# Test dims and nonnegativity of X,Y
@test(all(glrm.X .>= 0.0))
@test(all(glrm.Y .>= 0.0))
@test(size(glrm.X,1) == k)
@test(size(glrm.X,2) == m)
@test(size(glrm.Y,1) == k)
@test(size(glrm.Y,2) == n)

# validate against NMF.jl
A = [ 0.245774   0.481246   0.293614   0.528272   0.608255
      0.906146   0.0847498  0.963121   0.113728   0.552843
      0.269553   0.830981   0.067466   0.854045   0.170701
      0.781552   0.0302068  0.709045   0.578069   0.542792
      0.824889   0.538719   0.881363   0.229199   0.356273
      0.0530284  0.618962   0.96237    0.0877032  0.921746
      0.295122   0.626784   0.348475   0.299937   0.35043
      0.0499904  0.728344   0.573141   0.850758   0.425369
      0.872088   0.322181   0.903238   0.695946   0.706841
      0.542786   0.581426   0.0477561  0.601374   0.176598 ]

# Wt and H produced by NMF.jl
Wt = [0.25814 0.34329 0.24898 0.33489 0.35953 0.33798 0.23164 0.31705 0.43949 0.22484
      0.24305 0.0 0.71244 0.0 0.0 0.0 0.19987 0.49127 0.0 0.38994
      0.13531 0.0 0.0 0.0 0.0 0.94666 0.13278 0.26056 0.0 0.0]

H = [1.60738 1.42165 1.97295 1.47396 1.57709
     0.0 0.67822 0.0 0.6408 0.0
     0.0 0.23296 0.27065 0.0 0.35102]

# initialize glrm and check output
glrm.A = A
init_nndsvd!(glrm; scale=false)

@test(all(round(glrm.X,5) .== Wt))
@test(all(round(glrm.Y,5) .== H))

# Test with missing entries for A
obs = [(1,1),(1,3),(1,5),
       (2,1),(2,2),(2,4),
       (3,2),(3,3),(3,5),
       (4,1),(4,2),(4,5),
       (5,2),(5,3),(5,5),
       (6,1),(6,2),(6,5),
       (7,1),(7,3),(7,4),
       (8,2),(8,3),(8,4),
       (9,1),(9,3),(9,5),
       (10,3),(10,4),(10,5)]
glrm = GLRM(A,losses,r,r,k,obs=obs)
init_nndsvd!(glrm; max_iters=5)
