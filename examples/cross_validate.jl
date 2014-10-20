using LowRankModels, DataFrames

do_cv = false
do_cv_by_iter = false
do_reg_path = true

m,n,k = 100,100,3
A = randn(m,k)*randn(k,n) + k*sprandn(m,n,.01)
losses = fill(huber(),n)
r = quadreg(.1)
glrm = GLRM(A,losses,r,r,k)

if do_cv
    println("Computing cross validation error for each fold")
    train_error, test_error, train_glrms, test_glrms = cross_validate(glrm,5,Params(1,100,0,.001))
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_cv_by_iter
    println("Computing training and testing error as a function of iteration number")
    train_error, test_error = cv_by_iter(glrm)
    df = DataFrame(train_error = train_error, test_error = test_error)
end

if do_reg_path
    println("Computing regularization path")
    train_error, test_error, train_time, reg_params = 
    regularization_path(glrm, params=Params(1,50,.00001,.01), reg_params=logspace(2,-2,9))
    df = DataFrame(train_error = train_error, test_error = test_error,
                   train_time = train_time, reg_param = reg_params)
end

println(df)