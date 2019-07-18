using LowRankModels

# test losses in losses.jl
Random.seed!(1);

losses = [
QuadLoss(),
QuadLoss(10),
L1Loss(),
L1Loss(5.2),
HuberLoss(),
HuberLoss(4),
HuberLoss(3.1, crossover=3.2),
PeriodicLoss(2*pi),
PeriodicLoss(2*pi, 4),
PoissonLoss(20),
PoissonLoss(22,4.1),
OrdinalHingeLoss(1,10),
OrdinalHingeLoss(2,7,5),
LogisticLoss(),
LogisticLoss(0.2),
WeightedHingeLoss(),
WeightedHingeLoss(11),
WeightedHingeLoss(1.5, case_weight_ratio=4.3),
MultinomialLoss(4),
MultinomialLoss(6, .5),
# OrdisticLoss(5),
MultinomialOrdinalLoss(3)
] #tests what should be successful constructions

# TODO: do some bad constructions and test that they fail with catches
bad_losses = [
:(QuadLoss(10,RealDomain)),
:(HuberLoss(3.1, 3.2)),
:(PeriodicLoss(scale=2*pi)),
:(PeriodicLoss(2*pi, scale=4)),
:(PeriodicLoss())
]
for expression in bad_losses
	try
		eval(expression);
		println("test FAILED for $expression")
	catch
		println("test PASSED for $expression (failed to construct)")
	end
end

m,n,k = 1000, length(losses), 5;
d = embedding_dim(losses)
X_real, Y_real = 2*randn(m,k), 2*randn(k,d);
XY_real = X_real*Y_real;

# tests default imputations and implicit domains
# we can visually inspect the differences between A and A_real to make sure imputation is right
A = impute(losses, XY_real)

regscale = 1
yregs = Array(Regularizer, length(losses))
for i=1:length(losses)
	if typeof(losses[i]) == MultinomialOrdinalLoss ||
		typeof(losses[i]) == OrdisticLoss
		yregs[i] = OrdinalReg(QuadReg(regscale))
	else
		yregs[i] = QuadReg(regscale)
	end
end

# tests all the M-estimators with scale=false, offset=false
glrm = GLRM(A, losses, QuadReg(regscale), yregs, 5, scale=false, offset=false);

# interestingly adding an offset to a model with multidimensional ordinal data causes a segfault
# but let's test the offset for everything but ordinals
# oops we still get a segfault...
# tamecols = [typeof(losses[i]) !== MultinomialOrdinalLoss &&
# 			typeof(losses[i]) !== OrdisticLoss
# 			for i=1:length(losses)]
# glrm = GLRM(A[:, tamecols],
# 	losses[tamecols],
# 	QuadReg(regscale),
# 	yregs[tamecols],
# 	5, scale=false, offset=true)

# tests eval and grad
@time X,Y,ch = fit!(glrm);

# tests initialization
init_svd!(glrm)
