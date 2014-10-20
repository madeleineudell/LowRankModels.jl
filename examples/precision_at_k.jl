using LowRankModels, DataFrames

# boolean example with only entries greater than threshold t observed
# ie, censored data
# example with only entries greater than threshold t observed
m,n,k,ktrue = 10,10,1,1
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

(train_observed_features, train_observed_examples, test_observed_features,  test_observed_examples) = 
    get_train_and_test(obs, m, n, .2)

println(train_observed_features)
println(test_observed_features)

train_glrm = GLRM(B,train_observed_features, train_observed_examples,losses,r,r,k)

function precision_at_k(train_glrm::GLRM, test_observed_features; params=Params(), reg_params=logspace(2,-2,5), 
                        holdout_proportion=.1, verbose=true,
                        ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
    m,n = size(train_glrm.A)
    println(map(length, train_glrm.observed_features))
    println(map(length, test_observed_features))
    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_observed_features))
    println(ntest+ntrain)
    train_error = Array(Float64, length(reg_params))
    prec_at_k = Array(Float64, length(reg_params))
    solution = Array((Float64,Float64), length(reg_params))
    train_time = Array(Float64, length(reg_params))
    for iparam=1:length(reg_params)
        reg_param = reg_params[iparam]
        # evaluate train error
        if verbose println("fitting train GLRM for reg_param $reg_param") end
        train_glrm.rx.scale, train_glrm.ry.scale = reg_param, reg_param
        train_glrm.X, train_glrm.Y = randn(m,train_glrm.k), randn(train_glrm.k,n)
        X, Y, ch = fit!(train_glrm, params, ch, verbose=verbose)
        train_time[iparam] = ch.times[end]
        if verbose println("computing train and test error for reg_param $reg_param:") end
        train_error[iparam] = objective(train_glrm, X, Y, include_regularization=false)
        if verbose println("\ttrain error: $(train_error[iparam])") end
        # precision at k
        XY = X*Y
        q = sort(XY[:],rev=true)[ntest+ntrain] # the ntest+ntrain largest value in the model XY
        true_pos = 0; false_pos = 0
        for i=1:m
            for j=1:n
                if XY[i,j] >= q
                    if j in test_observed_features[i]
                        true_pos += 1
                    else
                        false_pos += 1
                    end
                end
            end
        end
        prec_at_k[iparam] = true_pos / (true_pos + false_pos)
        if verbose println("\prec_at_k:  $(prec_at_k[iparam])") end
        solution[iparam] = (sum(X)+sum(Y), sum(abs(X))+sum(abs(Y)))
        if verbose println("\tsum of solution, one norm of solution:  $(solution[iparam])") end
    end
    return train_error, prec_at_k, train_time, reg_params, solution
end

train_error, prec_at_k, train_time, reg_params, solution = 
    precision_at_k(train_glrm, test_observed_features, params=Params(1,200,.00001,.01), 
                                 reg_params=logspace(4,-2,7))   
df = DataFrame(train_error = train_error, prec_at_k = prec_at_k,
                   train_time = train_time, reg_param = reg_params, solution_1norm = [s[2] for s in solution])
println(df)

println(solution)