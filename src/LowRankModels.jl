module LowRankModels

using Compat

import Base: scale!, scale, show
import StatsBase.fit!

# define losses, regularizers, convergence history
include("domains.jl")
include("losses.jl")
include("impute_and_err.jl")
include("regularizers.jl")
include("convergence.jl")

# to use many processes to fit a model in shared memory, use shareglrm instead of glrm
#if nprocs()>1
#    include("shareglrm.jl")
#else
#    include("glrm.jl")
#end

# define basic data type
include("glrm.jl")

# fitting algorithms
include("fit.jl")
include("algorithms/proxgrad.jl")
include("algorithms/sparse_proxgrad.jl")

# initialization methods
include("rsvd.jl")
include("initialize.jl")

# fancy fun on top of low rank models
include("simple_glrms.jl")
include("cross_validate.jl")
include("fit_dataframe.jl")
# this takes to long to load for normal use
# include("plot.jl")

# utilities
include("utilities/conveniencemethods.jl")
include("utilities/deprecated.jl")

end # module
