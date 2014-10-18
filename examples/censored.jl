using LowRankModels, DataFrames

# boolean example with only entries greater than threshold t observed
# ie, censored data
# example with only entries greater than threshold t observed
m,n,k,ktrue = 500,500,5,2
m,n,k,ktrue = 500,500,5,2
t = 1
A = randn(m,ktrue)*randn(ktrue,n)
Anoisy = A + sprand(m,n,.01) # + .5*randn(m,n)
losses = fill(hinge(),n)
r = quadreg(.1)
obs = (Int64,Int64)[]
for i=1:m
	for j=1:n
		if Anoisy[i,j] > t
			push!(obs, (i,j))
		end
	end
end
Bnoisy = min(1,max(0,round(Anoisy)))
train_glrm = GLRM(Bnoisy,obs,losses,r,r,k)
# test on the whole dataset
B = min(1,max(0,round(A)))
test_glrm = GLRM(Bnoisy,losses,r,r,k)
#println(test_glrm.observed_features)

train_error, test_error, train_time, reg_params = 
	regularization_path(train_glrm, test_glrm, params=Params(1,50,.00001,.01), reg_params=logspace(2,-3,8))	
df = DataFrame(train_error = train_error, test_error = test_error,
		           train_time = train_time, reg_param = reg_params)
println(df)