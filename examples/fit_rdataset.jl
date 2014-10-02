using RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# fit it!
glrm, labels = GLRM(df,2)
X, Y, ch = fit(glrm)

# print results
println(ch)
println(labels)
println(Y)