using LowRankModels
using Test, Random, SparseArrays

using LinearAlgebra: norm

###############################################################
# verify basic functionality works
include("basic_functionality.jl")

###############################################################
# verify most losses, regularizers, and ways of calling glrm work
include("hello_world.jl")

################################################################################
# ScikitLearnBase test
import ScikitLearnBase

# Check that KMeans can correctly separate two non-overlapping Gaussians
Random.seed!(21)
gaussian1 = randn(100, 2) .+ 5.
gaussian2 = randn(50, 2) .- 10.
A = vcat(gaussian1, gaussian2)

model = ScikitLearnBase.fit!(LowRankModels.KMeans(), A)
@test Set(sum(ScikitLearnBase.transform(model, A), dims=1)) == Set([100, 50])

###############################################################
# test sparse and streaming functionality
include("sparse_test.jl")
include("streaming_test.jl")

###############################################################
# verify all examples run
include("../examples/runexamples.jl")

###############################################################
# verify all tests of specific loss functions run
include("prob_tests/runtests.jl")
