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
gfrm = GFRM(glrm; force=true, use_reg_scale = false)

# XXX for now...
train_error, test_error, train_glrms, test_glrms = cross_validate(gfrm, nfolds=5, params=PrismaParams(maxiter = 3))


U, ch = fit!(gfrm, PrismaParams(maxiter = 3))

# fit it!
glrmp = GLRM(gfrm, 3)
fit!(glrmp)

println("the difference between the model fit by the GLRM and the GFRM is")
@show(vecnorm(glrmp.X'*glrmp.Y - U)/vecnorm(U))

### test cross validation

do_cv_tests = true
do_cv = true
do_cv_by_iter = true
do_reg_path = true
do_plot = false

if do_cv && do_cv_tests
    println("Computing cross validation error for each fold")
	train_error, test_error, train_glrms, test_glrms = cross_validate(gfrm, nfolds=5, params=PrismaParams(maxiter = 3))
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_cv_by_iter && do_cv_tests
    println("Computing training and testing error as a function of iteration number")
    train_error, test_error = cv_by_iter(gfrm)
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_reg_path && do_cv_tests
    println("Computing regularization path")
    train_error, test_error, train_time, model_onenorm, reg_params = 
    regularization_path(gfrm, params=params, reg_params=logspace(2,-2,5))
    df = DataFrame(train_error = train_error, test_error = test_error,
                   train_time = train_time, model_onenorm=model_onenorm, reg_param = reg_params)
end