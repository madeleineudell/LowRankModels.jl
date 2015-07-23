using RDatasets
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# initialize
glrm, labels = GLRM(df,2)

println("Fitting model with random initialization")
X, Y, ch = fit!(glrm)
println("final objective is ", objective(glrm))

println("Fitting model with svd initialization; fit should be faster and final solution (slightly) better")
init_svd!(glrm)
X, Y, ch = fit!(glrm)
println("final objective is ", objective(glrm))