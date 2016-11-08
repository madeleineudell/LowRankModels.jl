# Supported domains: Real, Boolean, Ordinal, Periodic, Count

# The purpose of domains is to be able to sample over different possible values of `a` regardless of
# the loss that was used in the GLRM. The reason for doing this is to evaluate the performance of GLRMS.
# For instance, let's say we use PCA (QuadLoss losses) to model a binary data frame (not the best idea).
# In order to override the standard imputation with `sample(QuadLoss(), u)`, which assumes imputation over the reals,
# we can use `sample(BoolDomain(), QuadLoss(), u)` and see which of {-1,1} is best. The reason we want to be able to
# do this is to compare a baseline model (e.g. PCA) with a more logical model using heterogenous losses,
# yet still give each model the same amount of information regarding how imputation should be done.

# The domains themselves are defined in domains.jl

# In order to accomplish this we define a series of domains that describe how imputation should be performed over
# them. Each combination of domain and loss must have the following:
#	Methods:
#     `sample(D::my_Domain, l::my_loss_type, u::Float64) ::Float64`
#           Samples aᵤ from among the range of possible values of a. The range of
#			possible values of a should be implicitly or explicitly provided by `D`.
#			There should be an sample method for every combination of datatype and loss.

# DataTypes are assigned to each column of the data and are not part of the low-rank model itself, they just serve
# as a way to evaluate the performance of the low-rank model.

export sample, sample_missing

########################################## REALS ##########################################
# Real data can take values from ℜ

# l.scale should be 1/var
sample(D::RealDomain, l::QuadLoss, u::Float64) = u + randn()/sqrt(l.scale)

########################################## BOOLS ##########################################
# Boolean data should take values from {true, false}

function sample(D::BoolDomain, l::LogisticLoss, u::Float64)
	rand()<=(1/(1+exp(-u))) ? true : false
end

# generic method
# Evaluate w/ a=-1 and a=1 and see which is better according to that loss.
# This is fast and works for any loss.
function sample(D::BoolDomain, l::Loss, u::AbstractArray)
    prob = exp(-[evaluate(l, u, i) for i in (true, false)])
		return sample(WeightVec(prob))
end

########################################## ORDINALS ##########################################
# Ordinal data should take integer values ranging from `min` to `max`

# generic method
function sample(D::OrdinalDomain, l::Loss, u::AbstractArray)
    prob = exp(-[evaluate(l, u, i) for i in D.min:D.max])
		return sample(WeightVec(prob))
end

########################################## CATEGORICALS ##########################################
# Categorical data should take integer values ranging from 1 to `max`

function sample(D::CategoricalDomain, l::MultinomialLoss, u::Array{Float64})
	return sample(WeightVec(exp(u)))
end

# sample(D::CategoricalDomain, l::OvALoss, u::Array{Float64}) = ??

# generic method
function sample(D::CategoricalDomain, l::Loss, u::AbstractArray)
    prob = exp(-[evaluate(l, u, i) for i in D.min:D.max])
		return sample(WeightVec(prob))
end

########################################## PERIODIC ##########################################
# Periodic data can take values from ℜ, but given a period T, we should have error_metric(a,a+T) = 0

# Since periodic data can take any real value, we can use the real-valued imputation methods
# sample(D::PeriodicDomain, l::Loss, u::Float64) = ??

########################################## COUNTS ##########################################
# Count data can take values over ℕ, which we approximate as {0, 1, 2 ... `max_count`}

# Our approximation of ℕ is really an ordinal
sample(D::CountDomain, l::Loss, u::Float64) = sample(OrdinalDomain(0,D.max_count), l, u)

####################################################################################
# Use impute and error_metric over arrays
function sample{DomainSubtype<:Domain,LossSubtype<:Loss}(
			domains::Array{DomainSubtype,1},
			losses::Array{LossSubtype,1},
			U::Array{Float64,2}) # U = X'*Y
	m, d = size(U)
	n = length(losses)
	yidxs = get_yidxs(losses)
	A_sampled = Array(Number, (m, n));
	for f in 1:n
		for i in 1:m
			if length(yidxs[f]) > 1
				A_sampled[i,f] = sample(domains[f], losses[f], vec(U[i,yidxs[f]]))
			else
				A_sampled[i,f] = sample(domains[f], losses[f], U[i,yidxs[f]])
			end
		end
	end
	return A_sampled
end

# sample missing entries in A according to the fit model (X,Y)
function sample_missing(glrm::GLRM)
	m, d = size(U)
	n = length(losses)
	yidxs = get_yidxs(losses)
	A_sampled = copy(glrm.A);
	for f in 1:n
		for i in 1:m
			if !(i in glrm.observed_examples[f])
				if length(yidxs[f]) > 1
					A_sampled[i,f] = sample(domains[f], losses[f], vec(U[i,yidxs[f]]))
				else
					A_sampled[i,f] = sample(domains[f], losses[f], U[i,yidxs[f]])
				end
			end
		end
	end
	return A_sampled
end

# sample all entries in A according to the fit model (X,Y)
function sample(glrm::GLRM)
	m, d = size(U)
	n = length(losses)
	yidxs = get_yidxs(losses)
	A_sampled = copy(glrm.A);
	for f in 1:n
		for i in 1:m
			if length(yidxs[f]) > 1
				A_sampled[i,f] = sample(domains[f], losses[f], vec(U[i,yidxs[f]]))
			else
				A_sampled[i,f] = sample(domains[f], losses[f], U[i,yidxs[f]])
			end
		end
	end
	return A_sampled
end

function sample{LossSubtype<:Loss}(losses::Array{LossSubtype,1}, U::Array{Float64,2})
	domains = Domain[l.domain for l in losses]
	sample(domains, losses, U)
end
