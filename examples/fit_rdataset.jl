import RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# fit it!
glrm, labels = GLRM(df, 2, NaNs_to_NAs = true)
X, Y, ch = fit!(glrm)

# print results
println(ch)
println(labels)
println(Y)