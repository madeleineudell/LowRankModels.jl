# Supported data types: Real, Boolean, Ordinal, Periodic, Count

# The purpose of glrm_datatypes is to be able to impute over different possible values of `a` regardless of
# the loss that was used in the GLRM. The reason for doing this is to evaluate the performance of GLRMS.
# For instance, let's say we use PCA (quadratic losses) to model a binary data frame (not the best idea).
# In order to override the standard imputation with `impute(quadratic(), u)`, which assumes imputation over the reals,
# we can use `impute(gBool(), quadratic(), u)` and see which of {-1,1} is best. The reason we want to be able to 
# do this is to compare a baseline model (e.g. PCA) with a more logical model using heterogenous losses, 
# yet still give each model the same amount of information regarding how imputation should be done.

# In order to accomplish this we define a series of datatypes that describe how imputation should be performed over 
# them. Each datatype must have the following:
#	Methods: 
#     `impute(col_type::my_gDataType, l::my_loss_type, u::Float64) ::Float64`
#           Imputes aᵤ = argmin l(u,a) over the range of possible values of a. The range of 
#			possible values of a should be implicitly or explicitly provided by `col_type`.
#			There should be an impute method for every combination of datatype and loss.
#     `error_metric(col_type::my_gDataType, l::my_loss_type, u::Float64, a::Number) ::Float64`
#           First calls aᵤ = impute(l,u), then uses the type of `my_col_type` to pick a 
#			good measure of error- either 1-0 misclassification or squared difference.

# DataTypes are assigned to each column of the data and are not part of the low-rank model itself, they just serve
# as a way to evaluate the performance of the low-rank model.

export gDataType,
	   gReal, gBool, gOrdinal, gPeriodic, gCount,
	   impute, error_metric

# Error metrics for general use
squared_error(a_imputed::Float64, a::Number) = (a_imputed-a)^2
misclassification(a_imputed::Float64, a::Number) = float(!(a_imputed==a)) # return 0.0 if equal, 1.0 else

abstract gDataType

########################################## REALS ##########################################
# Real data can take values from ℜ
type gReal<:gDataType
end

impute(col_type::gReal, l::DiffLoss, u::Float64) = u # by the properties of any DiffLoss
impute(col_type::gReal, l::poisson, u::Float64) = exp(u)-1
impute(col_type::gReal, l::ordinal_hinge, u::Float64) = min(max(round(u),l.min),l.max)
impute(col_type::gReal, l::logistic, u::Float64) = error("Logistic loss always imputes either +∞ or -∞ given a∈ℜ")
function impute(col_type::gReal, l::weighted_hinge, u::Float64) 
	warn("It doesn't make sense to use hinge to impute data that can take values in ℜ")
	1/u
end

function error_metric(col_type::gReal, l::Loss, u::Float64, a::Number)
    a_imputed = impute(col_type, l, u)
    squared_error(a_imputed, a)
end

########################################## BOOLS ##########################################
# Boolean data should take values from {-1,1}
type gBool<:gDataType
end

# Evaluate w/ a=-1 and a=1 and see which is better according to that loss. 
# This is fast and works for any loss.
impute(col_type::gBool, l::Loss, u::Float64) = evaluate(l,u,-1.0)<evaluate(l,u,1.0) ? -1.0 : 1.0

function error_metric(col_type::gBool, l::Loss, u::Float64, a::Number)
	a_imputed = impute(col_type, l, u)
    misclassification(a_imputed, a)
end

########################################## ORDINALS ##########################################
# Ordinal data should take integer values ranging from `min` to `max`
type gOrdinal<:gDataType
	min::Int64
	max::Int64
end

impute(col_type::gOrdinal, l::DiffLoss, u::Float64) = min(max( round(u), col_type.min ), col_type.max)
impute(col_type::gOrdinal, l::poisson, u::Float64) = min(max( round(exp(u)-1), col_type.min ), col_type.max)
impute(col_type::gOrdinal, l::ordinal_hinge, u::Float64) = min(max( round(u), col_type.min), col_type.max)
impute(col_type::gOrdinal, l::logistic, u::Float64) = u>0 ? col_type.max : col_type.min
function impute(col_type::gOrdinal, l::weighted_hinge, u::Float64) 
	warn("It doesn't make sense to use hinge to impute ordinals")
	a_imputed = (u>0 ? ceil(1/u) : floor(1/u))
	min(max( round(a_imputed), col_type.min) ,col_type.max)
end

function error_metric(col_type::gOrdinal, l::Loss, u::Float64, a::Number)
    a_imputed = impute(col_type, l, u)
    squared_error(a_imputed, a)
end


########################################## PERIODIC ##########################################
# Periodic data can take values from ℜ, but given a period T, we should have error_metric(a,a+T) = 0
type gPeriodic<:gDataType
	T::Float64 # the period 
end

# Since periodic data can take any real value, we can use the real-valued imputation methods
impute(col_type::gPeriodic, l::Loss, u::Float64) = impute(gReal(), l, u)

# When imputing a periodic variable, we restrict ourselves to the domain [0,T]
pos_mod(T::Float64, x::Float64) = x>0 ? x%T : (x%T)+T # takes a value and finds its equivalent positive modulus
function error_metric(col_type::gPeriodic, l::Loss, u::Float64, a::Number)
    a_imputed = impute(col_type, l, u)
    # remap both a and a_imputed to [0,T] to check for a ≡ a_imputed
    squared_error(pos_mod(col_type.T,a_imputed), pos_mod(col_type.T,a)) 
end

########################################## COUNTS ##########################################
# Count data can take values over ℕ, which we approximate as {0, 1, 2 ... `max_count`}
type gCount<:gDataType
	max_count::Int64 # the biggest possible count
end

# Our approximation of ℕ is really an ordinal
impute(col_type::gOrdinal, l::Loss, u::Float64) = impute(gOrdinal(0,max_count), l, u)

function error_metric(col_type::gCount, l::Loss, u::Float64, a::Number)
    a_imputed = impute(l, u, col_type)
    squared_error(a_imputed, a)
end

####################################################################################
# To use these functions over arrays
function impute(types::Array{gDataType,1}, losses::Array{Loss,1}, A::Array{Float64,2})
	A_imputed = Array(Float64, size(A));
	m,n = size(A)
	for j in 1:n
		for i in 1:m
			A_imputed[i,j] = impute(types[j], losses[j], A[i,j]) # tests imputation
		end
	end 
	return A_imputed
end

function error_metric(types::Array{gDataType,1}, losses::Array{Loss,1}, 
					  U::Array{Float64,2}, A::Array{Float64,2} )
	err = Array(Float64, size(A));
	m,n = size(A)
	for j in 1:n
		for i in 1:m
			err[i,j] = error_metric(types[j], losses[j], U[i,j], A[i,j])
		end
	end
	return err
end

