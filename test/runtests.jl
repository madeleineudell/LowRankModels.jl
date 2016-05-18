using FullRankModels
using Base.Test

# write your own tests here
@test 1 == 1



################################################################################
# ScikitLearnBase test
using LowRankModels
import ScikitLearnBase

# Check that KMeans can correctly separate two non-overlapping Gaussians
srand(21)
gaussian1 = randn(100, 2) + 5
gaussian2 = randn(50, 2) - 10
A = vcat(gaussian1, gaussian2)

model = ScikitLearnBase.fit!(LowRankModels.KMeans(), A)
@test Set(sum(ScikitLearnBase.transform(model, A), 1)) == Set([100, 50])
