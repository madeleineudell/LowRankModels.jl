# Supported domains: Real, Boolean, Ordinal, Periodic, Count

# The purpose of domains is to be able to impute over different possible values of `a` regardless of
# the loss that was used in the GLRM. The reason for doing this is to evaluate the performance of GLRMS.
# For instance, let's say we use PCA (QuadLoss losses) to model a binary data frame (not the best idea).
# In order to override the standard imputation with `impute(QuadLoss(), u)`, which assumes imputation over the reals,
# we can use `impute(BoolDomain(), QuadLoss(), u)` and see which of {-1,1} is best. The reason we want to be able to
# do this is to compare a baseline model (e.g. PCA) with a more logical model using heterogenous losses,
# yet still give each model the same amount of information regarding how imputation should be done.

# In order to accomplish this we define a series of domains that tell imputation methods
# what values the data can take. The imputation methods are defined in impute_and_err.jl

# Domains should be assigned to each column of the data and are not part of the low-rank model itself.
# They serve as a way to evaluate the performance of the low-rank model.

export Domain, # the abstract type
	   RealDomain, BoolDomain, OrdinalDomain, PeriodicDomain, CountDomain, CategoricalDomain, # the domains
	   copy

abstract type Domain end

########################################## REALS ##########################################
# Real data can take values from ℜ
struct RealDomain<:Domain
end

########################################## BOOLS ##########################################
# Boolean data should take values from {true, false}
struct BoolDomain<:Domain
end

########################################## ORDINALS ##########################################
# Ordinal data should take integer values ranging from `min` to `max`
struct OrdinalDomain<:Domain
	min::Int
	max::Int
	function OrdinalDomain(min, max)
		if max - min < 2
			warn("The ordinal variable you've created is degenerate: it has only two levels. Consider using a Boolean variable instead; ordinal loss functions may have unexpected behavior on a degenerate ordinal domain.")
		end
		return new(min, max)
	end
end

########################################## ORDINALS ##########################################
# Categorical data should take integer values ranging from 1 to `max`
struct CategoricalDomain<:Domain
	min::Int
	max::Int
end
CategoricalDomain(m::Int) = CategoricalDomain(1,m)

########################################## PERIODIC ##########################################
# Periodic data can take values from ℜ, but given a period T, we should have error_metric(a,a+T) = 0
struct PeriodicDomain<:Domain
	T::Float64 # the period
end

########################################## COUNTS ##########################################
# Count data can take values over ℕ, which we approximate as {0, 1, 2 ... `max_count`}
struct CountDomain<:Domain
	max_count::Int # the biggest possible count
end
