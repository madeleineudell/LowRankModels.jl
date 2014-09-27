using RDatasets
include("autoencode_dataframe.jl")
# pick a data set
df = RDatasets.dataset("psych", "msq")
X,Y,labels,ch = autoencode_dataframe(df,2)
println(ch)
println(labels)
println(Y)