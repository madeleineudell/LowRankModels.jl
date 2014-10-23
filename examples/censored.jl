using LowRankModels, DataFrames

# boolean example with only entries greater than threshold t observed
# ie, censored data
# example with only entries greater than threshold t observed
m,n,k,ktrue = 500,500,1,1
A = rand(m,ktrue)*rand(ktrue,n)
println("max value of A is ",maximum(maximum(A))," which is less than $ktrue")
B = int(ktrue*rand(m,n) .>= A) # Bernoulli samples with probability proportional to A
losses = fill(quadratic(),n)
r = quadreg(.1)
obs = (Int64,Int64)[]
for i=1:m
    for j=1:n
        if B[i,j] == 1
            push!(obs, (i,j))
        end
    end
end
(train_observed_features, train_observed_examples,
    test_observed_features,  test_observed_examples) = 
    get_train_and_test(obs, m, n, .2)

train_glrm = GLRM(B,train_observed_features, train_observed_examples,losses,r,r,k)
test_glrm = GLRM(B,test_observed_features,  test_observed_examples,losses,r,r,k)

function censored_regularization_path(train_glrm::GLRM, test_glrm::GLRM; params=Params(), reg_params=logspace(2,-2,5), 
                                         holdout_proportion=.1, verbose=true,
                                         ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
    m,n = size(train_glrm.A)
    train_error = Array(Float64, length(reg_params))
    test_error = Array(Float64, length(reg_params))
    solution = Array((Float64,Float64), length(reg_params))
    train_time = Array(Float64, length(reg_params))
    for iparam=1:length(reg_params)
        reg_param = reg_params[iparam]
        # evaluate train and test error
        if verbose println("fitting train GLRM for reg_param $reg_param") end
        train_glrm.rx.scale, train_glrm.ry.scale = reg_param, reg_param
        train_glrm.X, train_glrm.Y = randn(m,train_glrm.k), randn(train_glrm.k,n)
        X, Y, ch = fit!(train_glrm, params, ch, verbose=verbose)
        train_time[iparam] = ch.times[end]
        if verbose println("computing train and test error for reg_param $reg_param:") end
        train_error[iparam] = objective(train_glrm, X, Y, include_regularization=false)
        if verbose println("\ttrain error: $(train_error[iparam])") end
        test_error[iparam] = objective(test_glrm, X, Y, include_regularization=false)
        if verbose println("\ttest error:  $(test_error[iparam])") end
        solution[iparam] = (sum(X)+sum(Y), sum(abs(X))+sum(abs(Y)))
        if verbose println("\tsum of solution, one norm of solution:  $(solution[iparam])") end
    end
    return train_error, test_error, train_time, reg_params, solution
end

train_error, test_error, train_time, reg_params, solution = 
    censored_regularization_path(train_glrm, test_glrm, params=Params(1,200,.00001,.01), 
                                 reg_params=logspace(4,-2,7))    
df = DataFrame(train_error = train_error, test_error = test_error,
                   train_time = train_time, reg_param = reg_params, solution_1norm = [s[2] for s in solution])
println(df)

println(solution)