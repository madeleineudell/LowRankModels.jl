using LowRankModels

# loss mutable structs to test
real_loss_mutable structs = [QuadLoss, HuberLoss]
bool_loss_mutable structs = [HingeLoss]
ordinal_loss_mutable structs = [OrdinalHingeLoss, BvSLoss]
categorical_loss_mutable structs = [MultinomialLoss, OvALoss]

#instantiate losses
ncat = 4 # maximum categorical levels
nord = 5 # maximum ordinal levels
real_losses = [l() for l in real_loss_mutable structs]
bool_losses = [l() for l in bool_loss_mutable structs]
ordinal_losses = [l(rand(3:nord)) for l in ordinal_loss_mutable structs]
categorical_losses = [l(rand(3:ncat)) for l in categorical_loss_mutable structs]
losses = [real_losses..., bool_losses..., ordinal_losses..., categorical_losses...]

# scale losses for different columns
for loss in losses
  mul!(loss, rand())
end

# regularizers to test
regularizers = [QuadReg(), OneReg(5), NonNegConstraint(), KSparseConstraint(2)]

m,n = length(regularizers), length(losses)

A_real = rand(m, length(real_losses))
A_bool = rand(Bool, m, length(bool_losses))
A_ord = rand(1:5, m, length(ordinal_losses))
A_cat = rand(1:3, m, length(categorical_losses))

# without saying "Any", upconverts to array of Floats
A = Any[A_real A_bool A_ord A_cat] # XXX upconverts to Array{Float64,2}

glrm = GLRM(A, losses, regularizers, QuadReg(), 2)
fit!(glrm)
println("successfully fit matrix")

### now fit data frame

using DataFrames
A = DataFrame(A)
for i=1:10
  A[rand(1:m), rand(1:n)] = NA
end
obs = observations(A)
glrm = GLRM(A, losses, QuadReg(), QuadReg(), 2, obs=obs)
fit!(glrm)
println("successfully fit dataframe")

# imputation and sampling
impute(glrm)
println("successfully imputed entries")
sample(glrm)
sample_missing(glrm)
println("successfully sampled from model")

### now fit sparse matrix

m, n = 100, 200
A = sprandn(m, n, .5)
glrm = GLRM(A, QuadLoss(), QuadReg(), QuadReg(), 5)
fit!(glrm)
println("successfully fit sparse GLRM")
