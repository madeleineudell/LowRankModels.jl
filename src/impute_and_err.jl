# Supported domains: Real, Boolean, Ordinal, Periodic, Count

# The purpose of domains is to be able to impute over different possible values of `a` regardless of
# the loss that was used in the GLRM. The reason for doing this is to evaluate the performance of GLRMS.
# For instance, let's say we use PCA (QuadLoss losses) to model a binary data frame (not the best idea).
# In order to override the standard imputation with `impute(QuadLoss(), u)`, which assumes imputation over the reals,
# we can use `impute(BoolDomain(), QuadLoss(), u)` and see which of {-1,1} is best. The reason we want to be able to
# do this is to compare a baseline model (e.g. PCA) with a more logical model using heterogenous losses,
# yet still give each model the same amount of information regarding how imputation should be done.

# The domains themselves are defined in domains.jl

# In order to accomplish this we define a series of domains that describe how imputation should be performed over
# them. Each combination of domain and loss must have the following:
#	Methods:
#     `impute(D::my_Domain, l::my_loss_type, u::Float64) ::Float64`
#           Imputes aᵤ = argmin l(u,a) over the range of possible values of a. The range of
#			possible values of a should be implicitly or explicitly provided by `D`.
#			There should be an impute method for every combination of datatype and loss.
#     `error_metric(D::my_Domain, l::my_loss_type, u::Float64, a::Number) ::Float64`
#           First calls aᵤ = impute(l,u), then uses the type of `my_D` to pick a
#			good measure of error- either 1-0 misclassification or squared difference.

# DataTypes are assigned to each column of the data and are not part of the low-rank model itself, they just serve
# as a way to evaluate the performance of the low-rank model.

export impute, error_metric, errors

# function for general use
roundcutoff(x,a::T,b::T) where T<:Number = T(min(max(round(x),a),b))

# Error metrics for general use
squared_error(a_imputed::Number, a::Number) = (a_imputed-a)^2
misclassification(a_imputed::T, a::T) where T = float(!(a_imputed==a)) # return 0.0 if equal, 1.0 else

# use the default loss domain imputation if no domain provided
impute(l::Loss, u::Float64) = impute(l.domain, l, u)

########################################## REALS ##########################################
# Real data can take values from ℜ

impute(D::RealDomain, l::DiffLoss, u::Float64) = u # by the properties of any DiffLoss
impute(D::RealDomain, l::PoissonLoss, u::Float64) = exp(u)
impute(D::RealDomain, l::OrdinalHingeLoss, u::Float64) = roundcutoff(u, l.min, l.max)
impute(D::RealDomain, l::LogisticLoss, u::Float64) = error("Logistic loss always imputes either +∞ or -∞ given a∈ℜ")
function impute(D::RealDomain, l::WeightedHingeLoss, u::Float64)
	warn("It doesn't make sense to use HingeLoss to impute data that can take values in ℜ")
	1/u
end

function error_metric(D::RealDomain, l::Loss, u::Float64, a::Number)
    a_imputed = impute(D, l, u)
    squared_error(a_imputed, a)
end

########################################## BOOLS ##########################################
# Boolean data should take values from {-1,1}

# sign of u
impute(D::BoolDomain, l::ClassificationLoss, u::Float64) = u>=0 ? true : false
# Evaluate w/ a=-1 and a=1 and see which is better according to that loss.
# This is fast and works for any loss.
impute(D::BoolDomain, l::Loss, u::Float64) = evaluate(l,u,false)<evaluate(l,u,true) ? false : true

function error_metric(D::BoolDomain, l::Loss, u::Float64, a::Number)
	a_imputed = impute(D, l, u)
    misclassification(a_imputed, a)
end

########################################## ORDINALS ##########################################
# Ordinal data should take integer values ranging from `min` to `max`

impute(D::OrdinalDomain, l::DiffLoss, u::Float64) = roundcutoff(u, D.min, D.max)
impute(D::OrdinalDomain, l::PoissonLoss, u::Float64) = roundcutoff(exp(u), D.min , D.max)
impute(D::OrdinalDomain, l::OrdinalHingeLoss, u::Float64) = roundcutoff(u, D.min, D.max)
impute(D::OrdinalDomain, l::LogisticLoss, u::Float64) = u>0 ? D.max : D.min
function impute(D::OrdinalDomain, l::WeightedHingeLoss, u::Float64)
	warn("It doesn't make sense to use HingeLoss to impute ordinals")
	a_imputed = (u>0 ? ceil(1/u) : floor(1/u))
	roundcutoff(a_imputed, D.min, D.max)
end
impute(D::OrdinalDomain, l::OrdisticLoss, u::AbstractArray) = argmin(u.^2)

# MultinomialOrdinalLoss
# l(u, a) = -log(p(u, a))
#            = u[1] + ... + u[a-1] - u[a] - ... - u[end] +
#              log(sum_{a'}(exp(u[1] + ... + u[a'-1] - u[a'] - ... - u[end])))
#
# so given u,
# the most probable value a is the index of the first
# positive entry of u
function impute(D::OrdinalDomain, l::MultinomialOrdinalLoss, u::AbstractArray)
	enforce_MNLOrdRules!(u)
	eu = exp.(u)
	p = [1-eu[1], -diff(eu)..., eu[end]]
	return argmax(p)
end

# generic method
function impute(D::OrdinalDomain, l::Loss, u::AbstractArray)
    (D.min:D.max)[argmin([evaluate(l, u, i) for i in D.min:D.max])]
end

function error_metric(D::OrdinalDomain, l::Loss, u::Float64, a::Number)
    a_imputed = impute(D, l, u)
    squared_error(a_imputed, a)
end

########################################## CATEGORICALS ##########################################
# Categorical data should take integer values ranging from 1 to `max`

impute(D::CategoricalDomain, l::MultinomialLoss, u::Array{Float64}) = argmax(u)
impute(D::CategoricalDomain, l::OvALoss, u::Array{Float64}) = argmax(u)

function error_metric(D::CategoricalDomain, l::Loss, u::Array{Float64}, a::Number)
    a_imputed = impute(D, l, u)
    misclassification(a_imputed, a)
end

########################################## PERIODIC ##########################################
# Periodic data can take values from ℜ, but given a period T, we should have error_metric(a,a+T) = 0

# Since periodic data can take any real value, we can use the real-valued imputation methods
impute(D::PeriodicDomain, l::Loss, u::Float64) = impute(RealDomain(), l, u)

# When imputing a periodic variable, we restrict ourselves to the domain [0,T]
pos_mod(T::Float64, x::Float64) = x>0 ? x%T : (x%T)+T # takes a value and finds its equivalent positive modulus
function error_metric(D::PeriodicDomain, l::Loss, u::Float64, a::Number)
    a_imputed = impute(D, l, u)
    # remap both a and a_imputed to [0,T] to check for a ≡ a_imputed
    squared_error(pos_mod(D.T,a_imputed), pos_mod(D.T,a))
end

########################################## COUNTS ##########################################
# Count data can take values over ℕ, which we approximate as {0, 1, 2 ... `max_count`}

# Our approximation of ℕ is really an ordinal
impute(D::CountDomain, l::Loss, u::Float64) = impute(OrdinalDomain(0,D.max_count), l, u)

function error_metric(D::CountDomain, l::Loss, u::Float64, a::Number)
    a_imputed = impute(D, l, u)
    squared_error(a_imputed, a)
end

####################################################################################
# Use impute and error_metric over arrays
function impute(domains::Array{DomainSubtype,1},
			losses::Array{LossSubtype,1},
			U::Array{Float64,2}) where {DomainSubtype<:Domain, LossSubtype<:Loss}
	m, d = size(U)
	n = length(losses)
	yidxs = get_yidxs(losses)
	A_imputed = Array{Number}(undef, (m, n));
	for f in 1:n
		for i in 1:m
			if length(yidxs[f]) > 1
				A_imputed[i,f] = impute(domains[f], losses[f], vec(U[i,yidxs[f]]))
			else
				A_imputed[i,f] = impute(domains[f], losses[f], U[i,yidxs[f]])
			end
		end
	end
	return A_imputed
end
function impute(losses::Array{LossSubtype,1}, U::Array{Float64,2}) where LossSubtype<:Loss
	domains = Domain[l.domain for l in losses]
	impute(domains, losses, U)
end

function errors(domains::Array{Domain,1}, losses::Array{Loss,1},
					  U::Array{Float64,2}, A::AbstractArray )
	err = zeros(size(A))
	m,n = size(A)
	for j in 1:n
		for i in 1:m
			err[i,j] = error_metric(domains[j], losses[j], U[i,j], A[i,j])
		end
	end
	return err
end
