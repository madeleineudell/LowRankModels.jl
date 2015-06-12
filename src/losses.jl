# Predefined loss functions
# You may also implement your own loss by subtyping the abstract type Loss.
#
# Losses must have the following:
#   Fields:
#     `scale::Float64`
#           This field represents a scalar weight assigned to the loss function: w*l(u,a)
#   Other fields may be also be included to encode parameters of the loss function, encode the range or  
#   set of possible values of the data, etc.
#
#   Methods:
#     `my_loss_type(args..., scale=1::Float64; kwargs...) ::my_loss_type`
#           Constructor for the loss type. The first few arguments are parameters for 
#           which there isn't a rational default (a loss may not have any of these).
#           The last positional argument should be the scale, which should default to 1.
#           Parameters besides the scale for which there are reasonable defaults should be
#           included as keyword arguments (there may be none).
#     `evaluate(l::my_loss_type, u::Float64, a::Number) ::Float64` 
#           Evaluates the function l(u,a) where u is the approximation of a
#     `grad(l::my_loss_type, u::Float64, a::Number) ::Float64`
#           Evaluates the gradient of the loss at the given point
#     `impute(l::my_loss_type, u::Float64) ::Float64`
#           Imputes aᵤ = argmin l(u,a) over the range of a. The range of a should either
#           be implicit, or derived from additional fields of my_loss_type
#     `error_metric(l::my_loss_type, u::Float64, a::Number) ::Float64`
#           First calls aᵤ = impute(l,u), then evaluates an "objective" error metric over 
#           aᵤ and a, which is either squared error where domain of a is real 
#           or 0-1 misclassification error if the domain of a is discrete
#   In addition, loss functions should preferably implement a method:
#     `M_estimator(l::my_loss_type, a::AbstractArray) ::Float64`
#           Finds uₒ = argmin ∑l(u,aᵢ) which is the best single estimate of the array a
#   If `M_estimator` is not implemented, a live optimization procedure will be used when this function is 
#   calledin order to compute loss function scalings. The live optimization may be slow, so an analytic 
#   implementation is preferable.


import Base.scale! 
import Optim.optimize
export Loss, # abstract types
       quadratic, weighted_hinge, hinge, logistic, ordinal_hinge, l1, huber, periodic, # concrete losses
       evaluate, grad, M_estimator, error_metric, impute, avgerror # methods on losses
       scale, scale!

abstract Loss
scale!(l::Loss, newscale::Number) = (l.scale = newscale; l)
scale(l::Loss) = l.scale

# This is the M-estimator for loss functions that don't have one defined. It's also useful for checking
# that the analytic M_estimators are correct. 
function M_estimator(l::Loss, a::AbstractArray, test=true) # pass in the third arg if you want to test
    # the function to optimize over
    f = u -> sum(map(ai->evaluate(l,u[1],ai), a)) # u is indexed because `optim` assumes input is a vector
    # the gradient of that function
    function g!(u::Vector, storage::Vector) # this is the format `optim` expects
        storage[1] = sum(map(ai->grad(l,u[1],ai), a))
    end
    m = optimize(f, g!, [median(a)], method=:l_bfgs).minimum[1]
end

function avgerror(l::Loss, a::AbstractArray)
    m = M_estimator(l,a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

# Error metrics for general use
squared_error(u::Float64, a::Number) = (u-a)^2
misclassification(u::Float64, a::Number) = float(u==a)

## Losses:

########################################## QUADRATIC ##########################################
# f: ℜxℜ -> ℜ
type quadratic<:Loss
    scale::Float64
end
quadratic() = quadratic(1)

evaluate(l::quadratic, u::Float64, a::Number) = l.scale*(u-a)^2

grad(l::quadratic, u::Float64, a::Number) = (u-a)*l.scale

impute(l::quadratic, u::Float64) = u

error_metric(l::quadratic, u::Float64, a::Number) = squared_error(u,a)

M_estimator(l::quadratic, a::AbstractArray) = mean(a)

########################################## L1 ##########################################
# f: ℜxℜ -> ℜ
type l1<:Loss
    scale::Float64
end
l1(scale=1.0) = l1(scale)

evaluate(l::l1, u::Float64, a::Number) = l.scale*abs(u-a)

grad(l::l1, u::Float64, a::Number) = sign(u-a)*l.scale

impute(l::l1, u::Float64) = u

error_metric(l::l1, u::Float64, a::Number) = squared_error(u,a)

M_estimator(l::l1, a::AbstractArray) = median(a)

########################################## HUBER ##########################################
# f: ℜxℜ -> ℜ
type huber<:Loss
    scale::Float64
    crossover::Float64 # where quadratic loss ends and linear loss begins; =1 for standard huber
end
huber(scale=1.0; crossover=1.0) = huber(scale, crossover)

function evaluate(l::huber, u::Float64, a::Number)
    abs(u-a) > l.crossover ? (abs(u-a) - l.crossover + l.crossover^2)*l.scale : (u-a)^2*l.scale
end

grad(l::huber,u::Float64,a::Number) = abs(u-a)>l.crossover ? sign(u-a)*l.scale : (u-a)*l.scale

impute(l::huber, u::Float64) = u

error_metric(l::huber, u::Float64, a::Number) = squared_error(u,a)

########################################## PERIODIC ##########################################
# f: ℜxℜ -> ℜ
# f(u,a) = w * (1 - cos((a-u)*(2*pi)/T))
# this measures how far away u and a are on a circle of circumference T. 
type periodic<:Loss
    T::Float64 # the length of the period
    scale::Float64
end
periodic(T) = periodic(T, 1)

evaluate(l::periodic, u::Float64, a::Number) = l.scale*(1-cos((a-u)*(2*pi)/l.T))

grad(l::periodic, u::Float64, a::Number) = -l.scale*((2*pi)/l.T)*sin((a-u)*(2*pi)/l.T)

function impute(l::periodic, u::Float64) 
    a = u%l.T
    a<0 ? a+l.T : a
end

error_metric(l::periodic, u::Float64, a::Number) = squared_error(impute(l,u),impute(l,a)) 
# impute a so that everything is properly matched

function M_estimator(l::periodic, a::AbstractArray)
    (l.T/(2*pi))*atan( sum(sin(2*pi*a/l.T)) / sum(cos(2*pi*a/l.T)) ) + l.T/2 # not kidding. 
    # this is the estimator, and there is a form that works with weighted measurements (aka a prior on a)
    # see: http://www.tandfonline.com/doi/pdf/10.1080/17442507308833101 eq. 5.2
end 

########################################## POISSON ##########################################
# f: ℜx(ℕ-0) -> ℜ
type poisson<:Loss
    scale::Float64
end
poisson(scale=1.0) = poisson(scale)

evaluate(l::poisson, u::Float64, a::Number) = exp(u) - a*u + a*log(a) - a

grad(l::poisson, u::Float64, a::Number) = exp(u) - a

function impute(l::poisson, u::Float64)
    a = round(exp(u))
    a>0 ? a : 1.0
end

error_metric(l::poisson, u::Float64, a::Number) = misclassification(impute(l,u),a)

########################################## LOGISTIC ##########################################
# f: ℜx{1,0} -> ℜ
type logistic<:Loss
    scale::Float64
end
logistic(scale=1.0) = logistic(scale)

evaluate(l::logistic, u::Float64, a::Number) = l.scale*log(1+exp(-a*u))

grad(l::logistic, u::Float64, a::Number) = -a*l.scale/(1+exp(a*u))

function M_estimator(l::logistic, a::AbstractArray)
    d, N = sum(a), length(N)
    log(N + d) - log(N - d) # very satisfying
end

impute(l::logistic, u::Float64) = u>=0 ? -1.0 : 1.0

error_metric(l::logistic, u::Float64, a::Number) = misclassification(impute(l,u),a)

########################################## ORDINAL HINGE ##########################################
# f: ℜx{min, min+1... max-1, max} -> ℜ
type ordinal_hinge<:Loss
    min::Integer
    max::Integer
    scale::Float64
end
ordinal_hinge(m1,m2,scale=1.0) = ordinal_hinge(m1,m2,scale)

function evaluate(l::ordinal_hinge, u::Float64, a::Number)
    if a == l.min 
        return l.scale*max(u-a,0)
    elseif a == l.max
        return l.scale*max(a-u,0)
    else
        return l.scale*abs(u-a)
    end    
end

function grad(l::ordinal_hinge, u::Float64, a::Number)
    if a == l.min 
        return max(sign(u-a), 0) * l.scale
    elseif a == l.max
        return min(sign(u-a), 0) * l.scale
    else
        return sign(u-a) * l.scale
    end
end

impute(l::ordinal_hinge, u::Float64) = min(max(round(u),l.min),l.max)

error_metric(l::ordinal_hinge, u::Float64, a::Number) = misclassification(impute(l,u),a)

M_estimator(l::ordinal_hinge, a::AbstractArray) = median(a)

########################################## WEIGHTED HINGE ##########################################
# f: ℜx{1,0} -> ℜ
# f(u,a) = {     w * max(0, u) for a = -1
#        = { c * w * max(0,-u) for a =  1
type weighted_hinge<:Loss
    scale::Float64
    case_weight_ratio::Float64 # >1 for trues to have more confidence than falses, <1 for opposite
end
weighted_hinge(scale=1.0; case_weight_ratio=1.0) = weighted_hinge(scale, case_weight_ratio)
hinge(scale=1.0) = weighted_hinge(scale) # the standard hinge is a case of this

function evaluate(l::weighted_hinge, u::Float64, a::Number)
    loss = l.scale*max(1-a*u, 0)
    if a>0 # if for whatever reason someone doesn't use properly coded variables...
        loss *= l.case_weight_ratio
    end
    return loss
end

function grad(l::weighted_hinge, u::Float64, a::Number)
    g = (a*u>=1 ? 0 : -a*l.scale)
    if a>0
        g *= l.case_weight_ratio
    end
    return g
end

impute(l::weighted_hinge, u::Float64) = evaluate(l,u,1.0)>evaluate(l,u,-1.0) ? -1.0 : 1.0

error_metric(l::weighted_hinge, u::Float64, a::Number) = misclassification(u,a)

function M_estimator(l::weighted_hinge, a::AbstractArray)
    r = length(a)/length(filter(x->x>0, a)) - 1 
    if l.case_weight_ratio > r
        m = 1.0
    elseif l.case_weight_ratio == r
        m = 0.0
    else
        m = -1.0
    end
end
