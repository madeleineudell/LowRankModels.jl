import RDatasets
import DataFrames: DataFrame
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

dd = DataFrame([df[s] for s in [:TOD, :Scale, :Vigorous, :Wakeful]])
dd[end] = (dd[end].==1)
datatypes = [:real, :cat, :ord, :bool]

# fit it!
glrm = GLRM(dd, 2, datatypes)
X, Y, ch = fit!(glrm)

# print results
println(ch.objective)
println(Y)

impute(glrm)