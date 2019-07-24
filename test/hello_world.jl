using LowRankModels, DataFrames, Random, SparseArrays
Random.seed!(0)

# loss types to test
real_loss_types = [QuadLoss, HuberLoss]
bool_loss_types = [HingeLoss]
ordinal_loss_types = [OrdinalHingeLoss, BvSLoss]
categorical_loss_types = [MultinomialLoss, OvALoss]

#instantiate losses
ncat = 4 # maximum categorical levels
nord = 5 # maximum ordinal levels
real_losses = [l() for l in real_loss_types]
bool_losses = [l() for l in bool_loss_types]
ordinal_losses = [l(rand(3:nord)) for l in ordinal_loss_types]
categorical_losses = [l(rand(3:ncat)) for l in categorical_loss_types]
losses = [real_losses..., bool_losses..., ordinal_losses..., categorical_losses...]
data_types = cat([:real for l in real_losses],
                  [:bool for l in bool_losses],
                  [:ord for l in ordinal_losses],
                  [:cat for l in categorical_losses],
                  dims=1)

# scale losses for different columns
for loss in losses
  mul!(loss, rand())
end

# regularizers to test
regularizers = [QuadReg(), OneReg(5), NonNegConstraint(), KSparseConstraint(2)]
# add more regularizers = more rows so the data isn't degenerate
regularizers = cat(regularizers, fill(QuadReg(), 10), dims=1)

m,n = length(regularizers), length(losses)

A_real = rand(m, length(real_losses))
A_bool = rand(Bool, m, length(bool_losses))
A_ord = rand(1:5, m, length(ordinal_losses))
A_cat = rand(1:3, m, length(categorical_losses))

# without saying "Any", upconverts to array of Floats
A = Any[A_real A_bool A_ord A_cat]

glrm = GLRM(A, losses, regularizers, QuadReg(), 2)
fit!(glrm, verbose=false)
println("successfully fit matrix")

Ω = [(rand(1:m), rand(1:n)) for iobs in 1:(5*max(m,n))] # observe some random entries, with replacement
glrm = GLRM(A, losses, regularizers, QuadReg(), 2, obs=Ω);
fit!(glrm, verbose=false)
println("successfully fit matrix with some entries unobserved")

### now fit data frame
A_sparse = sprandn(10, 10, .5)
df = NaNs_to_Missing!(DataFrame(Array(0 ./ A_sparse + A_sparse)))
# explicitly encoding missing
obs = observations(df)
glrm = GLRM(df, QuadLoss(), QuadReg(), QuadReg(), 2, obs=obs)
fit!(glrm, verbose=false)

# implicitly encoding missings from dataframe - this functionality has not been implemented for dataframes
# glrm = GLRM(df, QuadLoss(), QuadReg(), QuadReg(), 2)
# fit!(glrm, verbose=false)

# without specifying losses directly
glrm = GLRM(DataFrame(A), 3, data_types)
fit!(glrm, verbose=false)
println("successfully fit dataframe")

### imputation and sampling
impute(glrm)
println("successfully imputed entries")
sample(glrm)
sample_missing(glrm)
println("successfully sampled from model")

### now fit sparse matrix

m, n = 10, 10
sparseA = sprandn(m, n, .5)
glrm = GLRM(A, QuadLoss(), QuadReg(), QuadReg(), 5)
fit!(glrm, verbose=false)
println("successfully fit sparse GLRM")
