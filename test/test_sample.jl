using LowRankModels

A = sprandn(100,100,.5)
glrm = qpca(A, 3)

fit!(glrm)
A_sampled = sample(glrm)
A_sample_missing = sample_missing(glrm)

# tests
obs = !(A.==0)
@assert (A[obs]==A_sample_missing[obs])
