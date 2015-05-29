# Predefined loss functions and regularizers
# You may also implement your own loss or regularizer by subtyping 
# the abstract type Loss or Regularizer.
# Losses will need to have the methods `evaluate` and `grad` defined, 
# while regularizers should implement `evaluate` and `prox`. 
# For automatic scaling, losses should also implement `avgerror`.

import Base.scale! 
export Loss, Regularizer, # abstract types
       quadratic, weighted_hinge, hinge, logistic, ordinal_hinge, l1, huber, periodic, # concrete losses
       grad, evaluate, avgerror, # methods on losses
       quadreg, onereg, zeroreg, nonnegative, onesparse, unitonesparse, lastentry1, lastentry_unpenalized, # concrete regularizers
       prox, # methods on regularizers
       add_offset, # utilities
       scale, scale!

abstract Loss

# loss functions
scale!(l::Loss, newscale::Number) = (l.scale = newscale; l)
scale(l::Loss) = l.scale

## quadratic
type quadratic<:Loss
    scale::Float64
end
quadratic() = quadratic(1)
evaluate(l::quadratic,u::Float64,a::Number) = l.scale*(u-a)^2
grad(l::quadratic,u::Float64,a::Number) = (u-a)*l.scale

## l1
type l1<:Loss
    scale::Float64
end
l1() = l1(1)
evaluate(l::l1,u::Float64,a::Number) = l.scale*abs(u-a)
grad(l::l1,u::Float64,a::Number) = sign(u-a)*l.scale

## huber
type huber<:Loss
    scale::Float64
    crossover::Float64 # where quadratic loss ends and linear loss begins; =1 for standard huber
end
huber(scale) = huber(scale,1)
huber() = huber(1)
function evaluate(l::huber,u::Float64,a::Number)
    abs(u-a) > l.crossover ? (abs(u-a) - l.crossover + l.crossover^2)*l.scale : (u-a)^2*l.scale
end
grad(l::huber,u::Float64,a::Number) = abs(u-a)>l.crossover ? sign(u-a)*l.scale : (u-a)*l.scale

## poisson
type poisson<:Loss
    scale::Float64
end
poisson() = poisson(1)
evauate(l::poisson,u::Float64,a::Number) = exp(u) - a*u + a*log(a) - a
grad(l::poisson,u::Float64,a::Number) = exp(u) - a*u + a*log(a) - a

## logistic
type logistic<:Loss
    scale::Float64
end
logistic() = logistic(1)
evaluate(l::logistic,u::Float64,a::Number) = l.scale*log(1+exp(-a*u))
grad(l::logistic,u::Float64,a::Number) = -l.scale/(1+exp(a*u))

## ordinal hinge
type ordinal_hinge<:Loss
    min::Integer
    max::Integer
    scale::Float64
end
ordinal_hinge(m1,m2) = ordinal_hinge(m1,m2,1)
function grad(l::ordinal_hinge,u::Float64,a::Number)
    if a == l.min 
        return max(sign(u-a), 0) * l.scale
    elseif a == l.max
        return min(sign(u-a), 0) * l.scale
    else
        return sign(u-a) * l.scale
    end
end
function evaluate(l::ordinal_hinge,u::Float64,a::Number)
    if a == l.min 
        return l.scale*max(u-a,0)
    elseif a == l.max
        return l.scale*max(a-u,0)
    else
        return l.scale*abs(u-a)
    end    
end

## periodic
# f(u,a) = w * (1 - cos((a-u)*(2*pi)/T))
# this measures how far away u and a are on a circle of circumference T. 
type periodic<:Loss
    scale::Float64
    T::Integer # the length of the period
end
periodic(T; scale=1) = periodic(T, scale)
evaluate(l::periodic, u::Float64, a::Number) = l.scale*(1-cos((a-u)*(2*pi)/l.T))
grad(l::periodic, u::Float64, a::Number) = l.scale*((2*pi)/l.T)*sin((a-u)*(2*pi)/l.T)    

## weighted hinge
# f(u,a) = {     w * max(0, u) for a = -1
#        = { c * w * max(0,-u) for a =  1
type weighted_hinge<:Loss
    case_weight_ratio::Float64 # >1 for trues to have more confidence than falses, <1 for opposite
    scale::Float64
end
weighted_hinge(;case_weight_ratio=1, scale=1) = weighted_hinge(case_weight_ratio, scale)
hinge(;scale=1) = weighted_hinge(scale=scale) # the standard hinge is a case of this
function evaluate(l::weighted_hinge, u::Float64, a::Number)
    loss = l.scale*max(1+a*u, 0)
    if a>0 # if for whatever reason someone doesn't use properly coded variables...
        loss *= l.case_weight_ratio
    end
    return loss
end
evaluate(l::weighted_hinge, u::Float64, a::Bool) = evaluate(l, u, 2*a-1)
function grad(l::weighted_hinge, u::Float64, a::Number)
    g = (a*u>=1 ? 0 : -a*l.scale)
    if a>0
        g *= l.case_weight_ratio
    end
    return g
end
grad(l::weighted_hinge, u::Float64, a::Bool) = grad(l, u, 2*a-1)


# Useful functions for computing scalings
## minimum_offset (average error of l (a, offset))
function avgerror(a::AbstractArray, l::quadratic)
    m = mean(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::l1)
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::ordinal_hinge)
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::huber)
    # XXX this is not quite right --- mean is not necessarily the minimizer
    m = mean(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::periodic)
    m = (l.T/(2*pi))*atan( sum(sin(2*pi*a/l.T)) / sum(cos(2*pi*a/l.T)) ) + l.T/2# not kidding. 
    # this is the estimator, and there is a form that works with weighted measurements (aka a prior on a)
    # see: http://www.tandfonline.com/doi/pdf/10.1080/17442507308833101 eq. 5.2
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::weighted_hinge)
    r = length(a)/length(filter(x->x>0, a)) - 1 
    if l.case_weight_ratio > r
        m = 1.0
    elseif l.case_weight_ratio == r
        m = 0.0
    else
        m = -1.0
    end
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end
avgerror(a::AbstractArray{Bool,1}, l::weighted_hinge) = avgerror(2*a-1, l)

# regularizers
# regularizers r should have the method `prox` defined such that 
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)
abstract Regularizer

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); @simd for i=1:length(u) @inbounds u[i]=v[i] end; u)
scale(r::Regularizer) = r.scale
scale!(r::Regularizer, newscale::Number) = (r.scale = newscale; r)
scale!(rs::Array{Regularizer}, newscale::Number) = (for r in rs scale!(r, newscale) end; rs)

## quadratic regularization
type quadreg<:Regularizer
    scale::Float64
end
quadreg() = quadreg(1)
prox(r::quadreg,u::AbstractArray,alpha::Number) = 1/(1+alpha*r.scale/2)*u
prox!(r::quadreg,u::Array{Float64},alpha::Number) = scale!(u, 1/(1+alpha*r.scale/2))
evaluate(r::quadreg,a::AbstractArray) = r.scale*sum(a.^2)

## one norm regularization
type onereg<:Regularizer
    scale::Float64
end
onereg() = onereg(1)
prox(r::onereg,u::AbstractArray,alpha::Number) = max(u-alpha,0) + min(u+alpha,0)
evaluate(r::onereg,a::AbstractArray) = r.scale*sum(abs(a))

## no regularization
type zeroreg<:Regularizer
end
prox(r::zeroreg,u::AbstractArray,alpha::Number) = u
prox!(r::zeroreg,u::Array{Float64},alpha::Number) = u
evaluate(r::zeroreg,a::AbstractArray) = 0
scale(r::zeroreg) = 0
scale!(r::zeroreg, newscale::Number) = 0

## indicator of the nonnegative orthant 
## (enforces nonnegativity, eg for nonnegative matrix factorization)
type nonnegative<:Regularizer
end
prox(r::nonnegative,u::AbstractArray,alpha::Number) = broadcast(max,u,0)
prox!(r::nonnegative,u::Array{Float64},alpha::Number) = (@simd for i=1:length(u) @inbounds u[i] = max(u[i], 0) end; u)
evaluate(r::nonnegative,a::AbstractArray) = any(map(x->x<0,a)) ? Inf : 0
scale(r::nonnegative) = 1
scale!(r::nonnegative, newscale::Number) = 1

## indicator of the last entry being equal to 1
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry_unpenalized)
type lastentry1<:Regularizer
    r::Regularizer
end
prox(r::lastentry1,u::AbstractArray,alpha::Number) = [prox(r.r,u[1:end-1],alpha), 1]
prox!(r::lastentry1,u::Array{Float64},alpha::Number) = (prox!(r.r,u[1:end-1],alpha); u[end]=1; u)
evaluate(r::lastentry1,a::AbstractArray) = (a[end]==1 ? evaluate(r.r,a[1:end-1]) : Inf)
scale(r::lastentry1) = r.r.scale
scale!(r::lastentry1, newscale::Number) = (r.r.scale = newscale)

## makes the last entry unpenalized
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry1)
type lastentry_unpenalized<:Regularizer
    r::Regularizer
end
prox(r::lastentry_unpenalized,u::AbstractArray,alpha::Number) = [prox(r.r,u[1:end-1],alpha), u[end]]
prox!(r::lastentry_unpenalized,u::Array{Float64},alpha::Number) = (prox!(r.r,u[1:end-1],alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray) = evaluate(r.r,a[1:end-1])
scale(r::lastentry_unpenalized) = r.r.scale
scale!(r::lastentry_unpenalized, newscale::Number) = (r.r.scale = newscale)

## adds an offset to the model by modifying the regularizers
function add_offset(rx::Regularizer,ry::Regularizer)
    return lastentry1(rx), lastentry_unpenalized(ry)
end

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is nonzero, eg for orthogonal NNMF)
type onesparse<:Regularizer
end
prox(r::onesparse,u::AbstractArray,alpha::Number) = (idx = indmax(u); v=zeros(size(u)); v[idx]=u[idx]; v)
prox!(r::onesparse,u::Array,alpha::Number) = (idx = indmax(u); ui = u[idx]; scale!(u,0); u[idx]=ui; u)
evaluate(r::onesparse,a::AbstractArray) = sum(map(x->x>0,a)) <= 1 ? 0 : Inf 

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is 1 and all others are zero, eg for kmeans)
type unitonesparse<:Regularizer
end
prox(r::unitonesparse,u::AbstractArray,alpha::Number) = (idx = indmax(u); v=zeros(size(u)); v[idx]=1; v)
prox!(r::unitonesparse,u::Array,alpha::Number) = (idx = indmax(u); scale!(u,0); u[idx]=1; u)
evaluate(r::unitonesparse,a::AbstractArray) = ((sum(map(x->x>0,a)) <= 1 && sum(a)==1) ? 0 : Inf )