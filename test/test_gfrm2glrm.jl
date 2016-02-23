import RDatasets
import DataFrames: DataFrame
using LowRankModels
import FirstOrderOptimization: PrismaParams

# pick a data set
df = RDatasets.dataset("psych", "msq")

# just fit four of the columns, to try out all four data types
dd = DataFrame([df[s] for s in [:TOD, :Scale, :Vigorous, :Wakeful]])
dd[end] = (dd[end].==1)
datatypes = [:real, :cat, :ord, :bool]

# form model
glrm = GLRM(dd, 2, datatypes; scale=false, offset=false)

# full rank model
gfrm = GFRM(glrm; force=true, scale = false)
U, ch = fit!(gfrm, PrismaParams(maxiter = 3))

# fit it!
glrmp = GLRM(gfrm, 3)
fit!(glrmp)

@show(vecnorm(glrmp.X'*glrmp.Y - U)/vecnorm(U))