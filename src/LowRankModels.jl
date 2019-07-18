__precompile__()

module LowRankModels

using Printf
using SharedArrays
using SparseArrays
using Random

import LinearAlgebra: dot, norm, Diagonal, rmul!, mul!
import Base: show
import StatsBase: fit!, mode, mean, var, std

# define losses, regularizers, convergence history
include("domains.jl")
include("losses.jl")
include("impute_and_err.jl")
include("regularizers.jl")
include("convergence.jl")

# define basic data type(s)
include("glrm.jl")
include("shareglrm.jl")

# modify models (eg scaling and offsets) and evaluate fit
include("modify_glrm.jl")
include("evaluate_fit.jl")

# fitting algorithms
include("fit.jl")
if Threads.nthreads() > 1
  include("algorithms/proxgrad_multithread.jl")
else
  include("algorithms/proxgrad.jl")
end
include("algorithms/sparse_proxgrad.jl")
include("algorithms/quad_streaming.jl")

# initialization methods
include("rsvd.jl")
include("initialize.jl")

# fancy fun on top of low rank models
include("simple_glrms.jl")
include("cross_validate.jl")
include("fit_dataframe.jl")
include("sample.jl")
# this takes to long to load for normal use
# include("plot.jl")

# utilities
include("utilities/conveniencemethods.jl")
include("utilities/deprecated.jl")

# ScikitLearn.jl compatibility
include("scikitlearn.jl")

end # module
