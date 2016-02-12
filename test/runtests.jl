using LowRankModels

# just tests of syntactic correctness, no values tested
include("../examples/simple_glrms.jl")
include("../examples/fit_rdataset.jl")
include("../examples/cross_validate.jl")

# minimal values tested 
include("share_test.jl")

# actual tests
include("init_test.jl")
include("sparse_test.jl")
include("reg_test.jl")
include("loss_test.jl")

# some recovery tests for multidimensional glrms
include("mnl_test.jl")
include("ordistic_test.jl")
include("ordinallogit_test.jl")
