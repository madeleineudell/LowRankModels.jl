# Predefined loss functions
# You may also implement your own loss by subtyping the abstract type Loss.
#
# Losses must have the following:
#   Fields:
#     `scale::Float64`
#           This field represents a scalar weight assigned to the loss function: w*l(u,a)
#     `domain::natural_Domain`
#           The "natural" domain that the loss function was meant to handle. E.g. BoolDomain for LogisticLoss,
#           RealDomain for QuadLoss, etc.

#   Other fields may be also be included to encode parameters of the loss function, encode the range or
#   set of possible values of the data, etc.
#
#   Methods:
#     `my_loss_type(args..., scale=1.0::Float64;
#                   domain=natural_Domain(args[range]...), kwargs...) ::my_loss_type`
#           Constructor for the loss type. The first few arguments are parameters for
#           which there isn't a rational default (a loss may not need any of these).
#           The last positional argument should be the scale, which should default to 1.
#           There must be a default domain which is a Domain, which may take arguments from
#           the list of positional arguments. Parameters besides the scale for which there are
#           reasonable defaults should be included as keyword arguments (there may be none).
#     `evaluate(l::my_loss_type, u::Float64, a::Number) ::Float64`
#           Evaluates the function l(u,a) where u is the approximation of a
#     `grad(l::my_loss_type, u::Float64, a::Number) ::Float64`
#           Evaluates the gradient of the loss at the given point (u,a)

#   In addition, loss functions should preferably implement methods:
#     `M_estimator(l::my_loss_type, a::AbstractArray) ::Float64`
#           Finds uₒ = argmin ∑l(u,aᵢ) which is the best single estimate of the array `a`
#           If `M_estimator` is not implemented, a live optimization procedure will be used when this function is
#           called in order to compute loss function scalings. The live optimization may be slow, so an analytic
#           implementation is preferable.
#     `impute(d::Domain, l::my_loss_type, u::Array{Float64})` (in impute_and_err.jl)
#           Finds a = argmin l(u,a), the most likely value for an observation given a parameter u

import Base: scale!, *
import Optim.optimize
export Loss,
       DiffLoss, # a category of Losses
       QuadLoss, L1Loss, HuberLoss, QuantileLoss, # losses for predicting reals
       PoissonLoss, # losses for predicting integers
       HingeLoss, WeightedHingeLoss, LogisticLoss, # losses for predicting booleans
       OrdinalHingeLoss, OrdisticLoss, MultinomialOrdinalLoss, # losses for predicting ordinals
       MultinomialLoss, OvALoss, # losses for predicting nominals (categoricals)
       PeriodicLoss, # losses for predicting periodic variables
       evaluate, grad, M_estimator, # methods on losses
       avgerror, scale, scale!, *,
       embedding_dim, get_yidxs, datalevels

abstract Loss
# a DiffLoss is one in which l(u,a) = f(u-a) AND argmin f(x) = 0
# for example, QuadLoss(u,a)=(u-a)² and we can write f(x)=x² and x=u-a
abstract DiffLoss<:Loss

scale!(l::Loss, newscale::Number) = (l.scale = newscale; l)
scale(l::Loss) = l.scale
*(newscale::Number, l::Loss) = typeof(l)(scale(l)*newscale)

### embedding dimensions: mappings from losses/columns of A to columns of Y

# default number of columns
# number of columns is higher for multidimensional losses
embedding_dim(l::Loss) = 1
embedding_dim{LossSubtype<:Loss}(l::Array{LossSubtype,1}) = sum(map(embedding_dim, l))

# find spans of loss functions (for multidimensional losses)
function get_yidxs{LossSubtype<:Loss}(losses::Array{LossSubtype,1})
    n = length(losses)
    ds = map(embedding_dim, losses)
    d = sum(ds)
    featurestartidxs = cumsum(append!([1], ds))
    # find which columns of Y map to which columns of A (for multidimensional losses)
    @compat yidxs = Array(Union{Range{Int}, Int}, n)

    for f = 1:n
        if ds[f] == 1
            yidxs[f] = featurestartidxs[f]
        else
            yidxs[f] = featurestartidxs[f]:featurestartidxs[f]+ds[f]-1
        end
    end
    return yidxs
end

### M-estimators

# The following is the M-estimator for loss functions that don't have one defined. It's also useful
# for checking that the analytic M_estimators are correct. To make sure this method is called instead
# of the loss-specific method (should only be done to test), simply pass the third paramter `test`.
# e.g. M_estimator(l,a) will call the implementation for l, but M_estimator(l,a,"test") will call the
# general-purpose optimizing M_estimator.
function M_estimator(l::Loss, a::AbstractArray; test="test")
    # the function to optimize over
    f = u -> sum(map(ai->evaluate(l,u[1],ai), a)) # u is indexed because `optim` assumes input is a vector
    # the gradient of that function
    function g!(u::Vector, storage::Vector) # this is the format `optim` expects
        storage[1] = sum(map(ai->grad(l,u[1],ai), a))
    end
    m = optimize(f, g!, [median(a)], method=:l_bfgs).minimum[1]
end

# Uses uₒ = argmin ∑l(u,aᵢ) to find (1/n)*∑l(uₒ,aᵢ) which is the
# average error incurred by using the estimate uₒ for every aᵢ
function avgerror(l::Loss, a::AbstractArray)
    m = M_estimator(l,a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end


## Losses:

########################################## QUADRATIC ##########################################
# f: ℜxℜ -> ℜ
type QuadLoss<:DiffLoss
    scale::Float64
    domain::Domain
end
QuadLoss(scale=1.0::Float64; domain=RealDomain()) = QuadLoss(scale, domain)

evaluate(l::QuadLoss, u::Float64, a::Number) = l.scale*(u-a)^2

grad(l::QuadLoss, u::Float64, a::Number) = 2*(u-a)*l.scale

M_estimator(l::QuadLoss, a::AbstractArray) = mean(a)

########################################## L1 ##########################################
# f: ℜxℜ -> ℜ
type L1Loss<:DiffLoss
    scale::Float64
    domain::Domain
end
L1Loss(scale=1.0::Float64; domain=RealDomain()) = L1Loss(scale, domain)

evaluate(l::L1Loss, u::Float64, a::Number) = l.scale*abs(u-a)

grad(l::L1Loss, u::Float64, a::Number) = sign(u-a)*l.scale

M_estimator(l::L1Loss, a::AbstractArray) = median(a)

########################################## HUBER ##########################################
# f: ℜxℜ -> ℜ
type HuberLoss<:DiffLoss
    scale::Float64
    domain::Domain
    crossover::Float64 # where QuadLoss loss ends and linear loss begins; =1 for standard HuberLoss
end
HuberLoss(scale=1.0::Float64; domain=RealDomain(), crossover=1.0::Float64) = HuberLoss(scale, domain, crossover)

function evaluate(l::HuberLoss, u::Float64, a::Number)
    abs(u-a) > l.crossover ? (abs(u-a) - l.crossover + l.crossover^2)*l.scale : (u-a)^2*l.scale
end

grad(l::HuberLoss,u::Float64,a::Number) = abs(u-a)>l.crossover ? sign(u-a)*l.scale : (u-a)*l.scale

# M_estimator(l::HuberLoss, a::AbstractArray) = median(a) # a heuristic, not the true estimator.

########################################## QUANTILE ##########################################
# f: ℜxℜ -> ℜ
# define (u)_+ = max(u,0), (u)_- = max(-u,0) so (u)_+ + (u)_- = |u|
# f(u,a) = {    quantile (a - u)_+ + (1-quantile) (a - u)_-
# fits the `quantile`th quantile of the distribution
type QuantileLoss<:DiffLoss
    scale::Float64
    domain::Domain
    quantile::Float64 # fit the alphath quantile
end
QuantileLoss(scale=1.0::Float64; domain=RealDomain(), quantile=.5::Float64) = QuantileLoss(scale, domain, quantile)

function evaluate(l::QuantileLoss, u::Float64, a::Number)
    diff = a-u
    diff > 0  ?  l.scale * l.quantile * diff  :  - l.scale * (1-l.quantile) * diff
end

function grad(l::QuantileLoss,u::Float64,a::Number)
  diff = a-u
  diff > 0  ?  -l.scale * l.quantile  :  l.scale * (1-l.quantile)
end

M_estimator(l::QuantileLoss, a::AbstractArray) = quantile(a, l.quantile)

########################################## PERIODIC ##########################################
# f: ℜxℜ -> ℜ
# f(u,a) = w * (1 - cos((a-u)*(2*pi)/T))
# this measures how far away u and a are on a circle of circumference T.
type PeriodicLoss<:DiffLoss
    T::Float64 # the length of the period
    scale::Float64
    domain::Domain
end
PeriodicLoss(T, scale=1.0::Float64; domain=PeriodicDomain(T)) = PeriodicLoss(T, scale, domain)

evaluate(l::PeriodicLoss, u::Float64, a::Number) = l.scale*(1-cos((a-u)*(2*pi)/l.T))

grad(l::PeriodicLoss, u::Float64, a::Number) = -l.scale*((2*pi)/l.T)*sin((a-u)*(2*pi)/l.T)

function M_estimator(l::PeriodicLoss, a::AbstractArray{Float64})
    (l.T/(2*pi))*atan( sum(sin(2*pi*a/l.T)) / sum(cos(2*pi*a/l.T)) ) + l.T/2 # not kidding.
    # this is the estimator, and there is a form that works with weighted measurements (aka a prior on a)
    # see: http://www.tandfonline.com/doi/pdf/10.1080/17442507308833101 eq. 5.2
end

########################################## POISSON ##########################################
# f: ℜxℕ -> ℜ
# BEWARE:
# 1) this is a reparametrized poisson: we parametrize the mean as exp(u) so that u can take any real value and still produce a positive mean
# 2) THIS LOSS MAY CAUSE MODEL INSTABLITY AND DIFFICULTY FITTING.
type PoissonLoss<:Loss
    scale::Float64
    domain::Domain
end
PoissonLoss(max_count::Int, scale=1.0::Float64; domain=CountDomain(max_count)::Domain) = PoissonLoss(scale, domain)

function evaluate(l::PoissonLoss, u::Float64, a::Number)
    l.scale*(exp(u) - a*u) # in reality this should be: e^u - a*u + a*log(a) - a, but a*log(a) - a is constant wrt a!
end

grad(l::PoissonLoss, u::Float64, a::Number) = l.scale*(exp(u) - a)

M_estimator(l::PoissonLoss, a::AbstractArray) = log(mean(a))

########################################## ORDINAL HINGE ##########################################
# f: ℜx{min, min+1... max-1, max} -> ℜ
type OrdinalHingeLoss<:Loss
    min::Integer
    max::Integer
    scale::Float64
    domain::Domain
end
OrdinalHingeLoss(m1, m2, scale=1.0::Float64; domain=OrdinalDomain(m1,m2)) = OrdinalHingeLoss(m1,m2,scale,domain)
# this method should never be called directly but is needed to support copying
OrdinalHingeLoss() = OrdinalHingeLoss(1, 10, 1.0, OrdinalDomain(1,10))

function evaluate(l::OrdinalHingeLoss, u::Float64, a::Number)
    #a = round(a)
    if u > l.max-1
        # number of levels higher than true level
        n = min(floor(u), l.max-1) - a
        loss = n*(n+1)/2 + (n+1)*(u-l.max+1)
    elseif u > a
        # number of levels higher than true level
        n = min(floor(u), l.max) - a
        loss = n*(n+1)/2 + (n+1)*(u-floor(u))
    elseif u > l.min+1
        # number of levels lower than true level
        n = a - max(ceil(u), l.min+1)
        loss = n*(n+1)/2 + (n+1)*(ceil(u)-u)
    else
        # number of levels higher than true level
        n = a - max(ceil(u), l.min+1)
        loss = n*(n+1)/2 + (n+1)*(l.min+1-u)
    end
    return l.scale*loss
end

function grad(l::OrdinalHingeLoss, u::Float64, a::Number)
    #a = round(a)
    if u > a
        # number of levels higher than true level
        n = min(ceil(u), l.max) - a
        g = n
    else
        # number of levels lower than true level
        n = a - max(floor(u), l.min)
        g = -n
    end
    return l.scale*g
end

M_estimator(l::OrdinalHingeLoss, a::AbstractArray) = median(a)

########################################## LOGISTIC ##########################################
# f: ℜx{-1,1}-> ℜ
type LogisticLoss<:Loss
    scale::Float64
    domain::Domain
end
LogisticLoss(scale=1.0::Float64; domain=BoolDomain()) = LogisticLoss(scale, domain)

evaluate(l::LogisticLoss, u::Float64, a::Number) = l.scale*log(1+exp(-a*u))

grad(l::LogisticLoss, u::Float64, a::Number) = -a*l.scale/(1+exp(a*u))

function M_estimator(l::LogisticLoss, a::AbstractArray)
    d, N = sum(a), length(a)
    log(N + d) - log(N - d) # very satisfying
end

########################################## WEIGHTED HINGE ##########################################
# f: ℜx{-1,1} -> ℜ
# f(u,a) = {     w * max(1-a*u, 0) for a = -1
#        = { c * w * max(1-a*u, 0) for a =  1
type WeightedHingeLoss<:Loss
    scale::Float64
    domain::Domain
    case_weight_ratio::Float64 # >1 for trues to have more confidence than falses, <1 for opposite
end
WeightedHingeLoss(scale=1.0; domain=BoolDomain(), case_weight_ratio=1.0) =
    WeightedHingeLoss(scale, domain, case_weight_ratio)
HingeLoss(scale=1.0::Float64; kwargs...) = WeightedHingeLoss(scale; kwargs...) # the standard HingeLoss is a special case of WeightedHingeLoss

function evaluate(l::WeightedHingeLoss, u::Float64, a::Number)
    loss = l.scale*max(1-a*u, 0)
    if a>0 # if for whatever reason someone doesn't use properly coded variables...
        loss *= l.case_weight_ratio
    end
    return loss
end

function grad(l::WeightedHingeLoss, u::Float64, a::Number)
    g = (a*u>=1 ? 0 : -a*l.scale)
    if a>0
        g *= l.case_weight_ratio
    end
    return g
end

function M_estimator(l::WeightedHingeLoss, a::AbstractArray)
    r = length(a)/length(filter(x->x>0, a)) - 1
    if l.case_weight_ratio > r
        m = 1.0
    elseif l.case_weight_ratio == r
        m = 0.0
    else
        m = -1.0
    end
end

########################################## MULTINOMIAL ##########################################
# f: ℜx{1, 2, ..., max-1, max} -> ℜ
# f computes the (negative log likelihood of the) multinomial logit,
# often known as the softmax function
# f(u, a) = exp(u[a]) / (sum_{a'} exp(u[a']))
#         = 1         / (sum_{a'} exp(u[a'] - u[a]))
type MultinomialLoss<:Loss
    max::Integer
    scale::Float64
    domain::Domain
end
MultinomialLoss(m, scale=1.0::Float64; domain=CategoricalDomain(m)) = MultinomialLoss(m,scale,domain)
embedding_dim(l::MultinomialLoss) = l.max
datalevels(l::MultinomialLoss) = 1:l.max # levels are encoded as the numbers 1:l.max

# in Julia v0.4, argument u is a row vector (row slice of a matrix), which in julia is 2d
# function evaluate(l::MultinomialLoss, u::Array{Float64,2}, a::Int)
# this breaks compatibility with v0.4
function evaluate(l::MultinomialLoss, u::Array{Float64,1}, a::Int)
    sumexp = 0 # inverse likelihood of observation
    # computing soft max directly is numerically unstable
    # instead note logsumexp(a_j) = logsumexp(a_j - M) + M
    # and we'll pick a good big (but not too big) M
    M = maximum(u) - u[a] # prevents overflow
    for j in 1:length(u)
        sumexp += exp(u[j] - u[a] - M)
    end
    loss = log(sumexp) + M
    return l.scale*loss
end

# in Julia v0.4, argument u is a row vector (row slice of a matrix), which in julia is 2d
# function grad(l::MultinomialLoss, u::Array{Float64,2}, a::Int)
# this breaks compatibility with v0.4
function grad(l::MultinomialLoss, u::Array{Float64,1}, a::Int)
    g = zeros(size(u))
    # Using some nice algebra, you can show
    g[a] = -1
    # and g[b] = -1/sum_{a' \in S} exp(u[b] - u[a'])
    # the contribution of one observation to one entry of the gradient
    # is always between -1 and 0
    for j in 1:length(u)
        M = maximum(u) - u[j] # prevents overflow
        sumexp = 0
        for jp in 1:length(u)
            sumexp += exp(u[jp] - u[j] - M)
        end
        g[j] += exp(-M)/sumexp
    end
    return l.scale*g
end

## we'll compute it via a stochastic gradient method
## with fixed step size
function M_estimator(l::MultinomialLoss, a::AbstractArray)
    u = zeros(l.max)'
    for i = 1:length(a)
        ai = a[i]
        u -= .1*grad(l, u, ai)
    end
    return u
end

########################################## One vs All loss ##########################################
# f: ℜx{1, 2, ..., max-1, max} -> ℜ
type OvALoss<:Loss
    max::Integer
    bin_loss::Loss
    scale::Float64
    domain::Domain
end
OvALoss(m, scale=1.0::Float64; domain=CategoricalDomain(m), bin_loss::Loss=HingeLoss(scale)) = OvALoss(m,bin_loss,scale,domain)
embedding_dim(l::OvALoss) = l.max
datalevels(l::OvALoss) = 1:l.max # levels are encoded as the numbers 1:l.max

# in Julia v0.4, argument u is a row vector (row slice of a matrix), which in julia is 2d
# function evaluate(l::OvALoss, u::Array{Float64,2}, a::Int)
# this breaks compatibility with v0.4
function evaluate(l::OvALoss, u::Array{Float64,1}, a::Int)
    loss = 0
    for j in 1:length(u)
        loss += evaluate(l.bin_loss, u[j], a==j)
    end
    return l.scale*loss
end

# in Julia v0.4, argument u is a row vector (row slice of a matrix), which in julia is 2d
# function grad(l::OvALoss, u::Array{Float64,2}, a::Int)
# this breaks compatibility with v0.4
function grad(l::OvALoss, u::Array{Float64,1}, a::Int)
  g = zeros(length(u))
  for j in 1:length(u)
      g[j] = grad(l.bin_loss, u[j], a==j)
  end
  return l.scale*g
end

function M_estimator(l::OvALoss, a::AbstractArray)
    u = zeros(l.max)
    for i = 1:l.max
        u[i] = M_estimator(l.bin_loss, a.==i)
    end
    return u
end

########################################## ORDERED LOGISTIC ##########################################
# f: ℜx{1, 2, ..., max-1, max} -> ℜ
# f computes the (negative log likelihood of the) multinomial logit,
# often known as the softmax function
# f(u, a) = exp(u[a]) / (sum_{a'} exp(u[a']))
type OrdisticLoss<:Loss
    max::Integer
    scale::Float64
    domain::Domain
end
OrdisticLoss(m::Int, scale=1.0::Float64; domain=OrdinalDomain(1,m)) = OrdisticLoss(m,scale,domain)
embedding_dim(l::OrdisticLoss) = l.max
datalevels(l::OrdisticLoss) = 1:l.max # levels are encoded as the numbers 1:l.max

function evaluate(l::OrdisticLoss, u::Array{Float64,1}, a::Int)
    diffusquared = u[a]^2 .- u.^2
    M = maximum(diffusquared)
    invlik = sum(exp(diffusquared .- M))
    loss = M + log(invlik)
    return l.scale*loss
end

function grad(l::OrdisticLoss, u::Array{Float64,1}, a::Int)
    g = zeros(size(u))
    # Using some nice algebra, you can show
    g[a] = 2*u[a]
    sumexp = sum(map(j->exp(- u[j]^2), 1:length(u)))
    for j in 1:length(u)
        diffusquared = u[j]^2 .- u.^2
        M = maximum(diffusquared)
        invlik = sum(exp(diffusquared .- M))
        g[j] -= 2 * u[j] * exp(- M) / invlik
    end
    return l.scale*g
end

## we'll compute it via a stochastic gradient method
## with fixed step size
function M_estimator(l::OrdisticLoss, a::AbstractArray)
    u = zeros(l.max)'
    for i = 1:length(a)
        ai = a[i]
        u -= .1*grad(l, u, ai)
    end
    return u
end

#################### Multinomial Ordinal Logit #####################
# l: ℜ^{max-1} x {1, 2, ..., max-1, max} -> ℜ
# l computes the (negative log likelihood of the) multinomial ordinal logit.
#
# the length of the first argument u is one less than
# the number of levels of the second argument a,
# since the entries of u correspond to the division between each level
# and the one above it.
#
# To yield a sensible pdf, the entries of u should be increasing
# (b/c they're basically the -log of the cdf at the boundary between each level)
#
# The multinomial ordinal logit corresponds to a likelihood p with
# p(u, a > i) ~ exp(-u[i]), so
# p(u, a)     ~ exp(-u[1]) * ... * exp(-u[a-1]) * exp(u[a]) * ... * exp(u[end])
#             = exp(- u[1] - ... - u[a-1] + u[a] + ... + u[end])
# and normalizing,
# p(u, a)     = p(u, a) / sum_{a'} p(u, a')
#
# So l(u, a) = -log(p(u, a))
#            = u[1] + ... + u[a-1] - u[a] - ... - u[end] +
#              log(sum_{a'}(exp(u[1] + ... + u[a'-1] - u[a'] - ... - u[end])))
#
# Inspection of this loss function confirms that given u,
# the most probable value a is the index of the first
# positive entry of u

type MultinomialOrdinalLoss<:Loss
    max::Integer
    scale::Float64
    domain::Domain
end
MultinomialOrdinalLoss(m::Int, scale=1.0::Float64; domain=OrdinalDomain(1,m)) = MultinomialOrdinalLoss(m,scale,domain)
embedding_dim(l::MultinomialOrdinalLoss) = l.max - 1
datalevels(l::MultinomialOrdinalLoss) = 1:l.max # levels are encoded as the numbers 1:l.max

# argument u is a row vector (row slice of a matrix), which in julia is 2d
# todo: increase numerical stability
function evaluate(l::MultinomialOrdinalLoss, u::Array{Float64,1}, a::Int)
    diffs = zeros(l.max)
    for i=1:l.max
        diffs[i] = sum(u[1:i-1]) - sum(u[i:end])
    end
    M = maximum(diffs) # enhanced numerical stability: exp of small numbers only
    expdiffs = exp(diffs .- M)
    loss = diffs[a] + M + log(sum(expdiffs))
    return l.scale*loss
end

# argument u is a row vector (row slice of a matrix), which in julia is 2d
function grad(l::MultinomialOrdinalLoss, u::Array{Float64,1}, a::Int)
    signedsums = Array(Float64, l.max-1, l.max)
    for i=1:l.max-1
        for j=1:l.max
            signedsums[i,j] = i<j ? 1 : -1
        end
    end
    g = signedsums[:,a]
    diffs = signedsums'*u
    M = maximum(diffs) # enhanced numerical stability: exp of small numbers only
    expdiffs = exp(diffs .- M)
    sumexpdiffs = sum(expdiffs)
    for i=1:l.max
        g += expdiffs[i]/sumexpdiffs*signedsums[:,i]
    end
    return l.scale*g
end

## we'll compute it via a stochastic gradient method
## with fixed step size
## (we don't need a hyper accurate estimate for this)
function M_estimator(l::MultinomialOrdinalLoss, a::AbstractArray)
    u = zeros(l.max-1)'
    for i = 1:length(a)
        ai = a[i]
        u -= .1*grad(l, u, ai)
    end
    return u
end


### convenience methods for evaluating and computing gradients on vectorized arguments

function evaluate(l::Loss, u::Array{Float64,1}, a::Array{Float64,1})
  @assert size(u) == size(a)
  out = 0
  for i=1:length(a)
    out += evaluate(l, u[i], a[i])
  end
  return out
end
function evaluate(l::Loss, u::Array{Float64,2}, a::Array{Float64,1})
  @assert size(u,1) == size(a)
  out = 0
  for i=1:length(a)
    out += evaluate(l, u[i,:], a[i])
  end
  return out
end

function grad(l::Loss, u::Array{Float64,1}, a::Array{Float64,1})
  @assert size(u) == size(a)
  mygrad = zeros(size(u))
  for i=1:length(a)
    mygrad[i] = grad(l, u[i], a[i])
  end
  return mygrad
end
function grad(l::Loss, u::Array{Float64,2}, a::Array{Float64,1})
  @assert size(u,1) == size(a)
  mygrad = zeros(size(u))
  for i=1:length(a)
    mygrad[i] = grad(l, u[i,:], a[i])
  end
  return mygrad
end
