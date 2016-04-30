import RDatasets
import DataFrames: DataFrame
using LowRankModels

# pick a data set
df = RDatasets.dataset("psych", "msq")

# make a GLRM on the whole dataframe using type imputation
auto_glrm = GLRM(df, 3)

# now we'll try it without type imputation
# we'll just fit four of the columns, to try out all four data types
dd = DataFrame([df[s] for s in [:TOD, :Scale, :Vigorous, :Wakeful]])
dd[end] = (dd[end].==1)
datatypes = [:real, :cat, :ord, :bool]

# fit it!
glrm = GLRM(dd, 2, datatypes)

println("initializing")
init_svd!(glrm)

println("fitting")
X, Y, ch = fit!(glrm)

# print results
println(ch.objective)

println("imputing")
impute(glrm)

println("crossvalidating")
cross_validate(glrm, do_obs_check=false, init=init_svd!)