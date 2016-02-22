module LowRankModels

using Compat

import Base: scale!, scale, show
import StatsBase: fit!, mode

# define losses, regularizers, convergence history
include("domains.jl")
include("losses.jl")
include("impute_and_err.jl")
include("regularizers.jl")
include("convergence.jl")

# define basic data type(s)
include("glrm.jl")
include("gfrm.jl")
include("shareglrm.jl")

# modify models (eg scaling and offsets) and evaluate fit
include("modify_glrm.jl")
include("evaluate_fit.jl")

# fitting algorithms
include("fit.jl")
include("algorithms/proxgrad.jl")
include("algorithms/sparse_proxgrad.jl")
include("algorithms/parallel_proxgrad.jl")
include("algorithms/prisma.jl")

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
