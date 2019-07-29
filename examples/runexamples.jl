using RDatasets

include("censored.jl")
include("cross_validate.jl")
#=
Initialize example fails because we're passing -1 as the third argument to
```
evaluate(::MultinomialOrdinalLoss, ::Array{Float64,1}, ::Int32)
```
src/evaluate_fit.jl:15
...
initialize.jl:11
=#
# include("initialize.jl")
include("precision_at_k.jl")
include("simple_glrms.jl")
include("fit_rdataset.jl")
