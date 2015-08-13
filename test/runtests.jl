using LowRankModels

# just tests of syntactic correctness, no values tested
include("../examples/simple_glrms.jl")

# actual tests
include("init_test.jl")
include("sparse_test.jl")