# Predefined loss functions and regularizers
# You may also implement your own loss or regularizer by subtyping 
# the abstract type Loss or Regularizer.
# Losses will need to have the methods `evaluate` and `grad` defined, 
# while regularizers should implement `evaluate` and `prox`. 
# For automatic scaling, losses should also implement `avgerror`.

import Base.scale!
export Loss, Regularizer, # abstract types
       quadratic, hinge, ordinal_hinge, l1, huber, # concrete losses
       grad, evaluate, avgerror, # methods on losses
       quadreg, onereg, zeroreg, nonnegative, onesparse, unitonesparse, lastentry1, lastentry_unpenalized, # concrete regularizers
       prox, # methods on regularizers
       add_offset, # utilities
       scale, scale!

abstract Loss

# loss functions
scale!(l::Loss, newscale::Number) = (l.scale = newscale; l)

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

## hinge
type hinge<:Loss
    scale::Float64
end
hinge() = hinge(1)
evaluate(l::hinge,u::Float64,a::Number) = l.scale*max(1-a*u,0)
grad(l::hinge,u::Float64,a::Number) = hinge_grad(l.scale,u,a)
hinge_grad(scale,u::Float64,a::Number) = a*u>=1 ? 0 : -a*scale
hinge_grad(scale,u::Float64,a::Bool) = (2*a-1)*u>=1 ? 0 : -(2*a-1)*scale

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

function avgerror(a::AbstractArray, l::hinge)
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function avgerror(a::AbstractArray, l::huber)
    # XXX this is not quite right --- mean is not necessarily the minimizer
    m = mean(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

# regularizers
# regularizers r should have the method `prox` defined such that 
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)
abstract Regularizer

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); @simd for i=1:length(u) @inbounds u[i]=v[i] end; u)
scale(r::Regularizer) = r.scale
scale!(r::Regularizer, newscale::Number) = (r.scale = newscale; r)

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
prox(r::onereg,u::AbstractArray,alpha::Number) = max(u-as,0) + min(u+as,0)
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