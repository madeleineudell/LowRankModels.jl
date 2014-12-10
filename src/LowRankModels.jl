module LowRankModels

# define losses, regularizers, convergence history
include("loss_and_reg.jl")
include("convergence.jl")

## to use many processes to fit a model in shared memory, uncomment line 8 and comment line 9
#include("shareglrm.jl")
include("glrm.jl")

# fancy fun on top of low rank models
include("cross_validate.jl")
include("fit_dataframe.jl")
#include("plot.jl")

end # module
