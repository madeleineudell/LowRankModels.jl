import RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# fit it!
glrm, labels = GLRM(df, 2, NaNs_to_NAs = true, scale = false, offset=false)
gfrm = GFRM(glrm)
X, Y, ch = fit!(gfrm)
glrmp = GLRM(gfrm)
fit!(glrmp)

@show(vecnorm(glrmp.X'*glrmp.Y - gfrm.W)/vecnorm(gfrm.W))

# print results
println(ch)
println(labels)
println(Y)