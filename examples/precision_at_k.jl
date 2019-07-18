using DataFrames, LowRankModels

# boolean example with only entries greater than threshold t observed
# ie, censored data
# example with only entries greater than threshold t observed
m,n,k,ktrue = 100,100,1,1
A = rand(m,ktrue)*rand(ktrue,n)
println("max value of A is ",maximum(maximum(A))," which is less than $ktrue")
B = round.(Int, ktrue*rand(m,n) .>= A) # Bernoulli samples with probability proportional to A
losses = fill(QuadLoss(),n)
r = QuadReg(.1)
obs = Array{Tuple{Int,Int}}(undef, 0)
for i=1:m
    for j=1:n
        if B[i,j] == 1
            push!(obs, (i,j))
        end
    end
end

(train_observed_features, train_observed_examples, test_observed_features,  test_observed_examples) =
    get_train_and_test(obs, m, n, .2)

train_glrm = GLRM(B,losses,r,r,k, observed_features=train_observed_features, observed_examples=train_observed_examples)

train_error, test_error, prec_at_k, train_time, reg_params, solution =
    precision_at_k(train_glrm, test_observed_features,
    					params=Params(1, max_iter=200, abs_tol=0.00001, min_stepsize=0.01),
                                 reg_params=exp10.(range(2,step=-2,length=9)))
df = DataFrame(train_error = train_error, prec_at_k = prec_at_k,
                   train_time = train_time, reg_param = reg_params, solution_1norm = [s[2] for s in solution]);
println(df)
