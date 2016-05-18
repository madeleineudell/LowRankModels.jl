using DataFrames, LowRankModels

println("cross validation example")
srand(5)

do_cv = true
do_cv_by_iter = true
do_reg_path = true

m,n,k = 50,50,3
A = randn(m,k)*randn(k,n) + k*sprandn(m,n,.05)
losses = fill(HuberLoss(),n)
r = QuadReg(.1)
glrm = GLRM(A,losses,r,r,k+2)

if do_cv
    println("Computing cross validation error for each fold")
    params = Params(1.0, max_iter=100, abs_tol=0.0, min_stepsize=.001)
    train_error, test_error, train_glrms, test_glrms = cross_validate(glrm, nfolds=5, params=params)
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_cv_by_iter
    println("Computing training and testing error as a function of iteration number")
    train_error, test_error = cv_by_iter(glrm)
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_reg_path
    println("Computing regularization path")
    params = Params(1.0, max_iter=50, abs_tol=.00001, min_stepsize=.01)
    train_error, test_error, train_time, reg_params = 
    regularization_path(glrm, params=params, reg_params=logspace(2,-2,15))
    df = DataFrame(train_error = train_error, test_error = test_error,
                   train_time = train_time, reg_param = reg_params)
	if do_plot 
		p = plot(df, :reg_param, [:train_error, :test_error]; scale = :log, filename = None)
	end
end

println(df)