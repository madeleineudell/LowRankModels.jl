using RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# fit it!
X,Y,labels,ch = fit(GLRM(df,2))

# print results
println(ch)
println(labels)
println(Y)