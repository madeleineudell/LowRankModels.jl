using RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# encode it!
X,Y,labels,ch = autoencode_dataframe(df,2)

# print results
println(ch)
println(labels)
println(Y)