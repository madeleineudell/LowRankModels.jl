# Predefined loss functions and regularizers
# You may also implement your own loss or regularizer by subtyping 
# the abstract type Loss or Regularizer.
# Losses will need to have the methods `evaluate` and `grad` defined, 
# while regularizers should implement `evaluate` and `prox`. 
# For automatic scaling, losses should also implement `avgerror`.

export Loss, Regularizer, # abstract types
       quadratic, hinge, ordinal_hinge, l1, huber, # concrete losses
       grad, evaluate, avgerror, # methods on losses
       quadreg, onereg, zeroreg, nonnegative, onesparse, lastentry1, lastentry_unpenalized, # concrete regularizers
       prox, # methods on regularizers
       add_offset, equilibrate_variance! # utilities

abstract Loss

# loss functions
type quadratic<:Loss
    scale
end
quadratic() = quadratic(1)
type l1<:Loss
    scale
end
l1() = l1(1)
type hinge<:Loss
    scale
end
hinge() = hinge(1)
type ordinal_hinge<:Loss
    min::Integer
    max::Integer
    scale
end
ordinal_hinge(m1,m2) = ordinal_hinge(m1,m2,1)
type huber<:Loss
    scale
    crossover # where quadratic loss ends and linear loss begins; =1 for standard huber
end
huber(scale) = huber(scale,1)
huber() = huber(1)

## gradients of loss functions
grad(l::quadratic,u::Number,a::Number) = (u-a)*l.scale

grad(l::l1,u::Number,a::Number) = sign(u-a)*l.scale

grad(l::hinge,u::Number,a::Number) = hinge_grad(l.scale,u,a)
hinge_grad(scale,u,a::Number) = a*u>=1 ? 0 : -a*scale
hinge_grad(scale,u,a::Bool) = (2*a-1)*u>=1 ? 0 : -(2*a-1)*scale

function grad(l::ordinal_hinge,u::Number,a::Number)
    if a == l.min 
        if u>a
            return l.scale
        else 
            return 0
        end
    elseif a == l.max
        if u<a
            return -l.scale
        else
            return 0
        end
    else
        return sign(u-a) * l.scale
    end
end

grad(l::huber,u::Number,a::Number) = abs(u-a)>l.crossover ? sign(u-a)*l.scale : (u-a)*l.scale

## evaluating loss functions
evaluate(l::quadratic,u::Number,a::Number) = l.scale*(u-a)^2
evaluate(l::l1,u::Number,a::Number) = l.scale*abs(u-a)
evaluate(l::hinge,u::Number,a::Number) = l.scale*max(1-a*u,0)

function evaluate(l::ordinal_hinge,u::Number,a::Number)
    if a == l.min 
        return l.scale*max(u-a,0)
    elseif a == l.max
        return l.scale*max(a-u,0)
    else
        return l.scale*abs(u-a)
    end    
end
function evaluate(l::huber,u::Number,a::Number)
    abs(u-a) > l.crossover ? (abs(u-a) - l.crossover + l.crossover^2)*l.scale : (u-a)^2*l.scale
end

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
    # XXX this is not quite right
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

# regularizers
# regularizers r should have the method `prox` defined such that 
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)

abstract Regularizer

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); for i=1:length(u) u[i]=v[i] end; u)

## quadratic regularization
type quadreg<:Regularizer
    scale
end
quadreg() = quadreg(1)
prox(r::quadreg,u::AbstractArray,alpha::Number) = 1/(1+alpha*r.scale/2)*u
prox!(r::quadreg,u::AbstractArray,alpha::Number) = scale!(u, 1/(1+alpha*r.scale/2))
evaluate(r::quadreg,a::AbstractArray) = r.scale*sum(a.^2)

## one norm regularization
type onereg<:Regularizer
    scale
end
onereg() = onereg(1)
prox(r::onereg,u::AbstractArray,alpha::Number) = error("Not Implemented")
evaluate(r::onereg,a::AbstractArray) = r.scale*sum(abs(a))

## no regularization
type zeroreg<:Regularizer
end
prox(r::zeroreg,u::AbstractArray,alpha::Number) = u
prox!(r::zeroreg,u::AbstractArray,alpha::Number) = u
evaluate(r::zeroreg,a::AbstractArray) = 0

## indicator of the nonnegative orthant 
## (enforces nonnegativity, eg for nonnegative matrix factorization)
type nonnegative<:Regularizer
end
prox(r::nonnegative,u::AbstractArray,alpha::Number) = broadcast(max,u,0)
evaluate(r::nonnegative,a::AbstractArray) = any(map(x->x<0,a)) ? Inf : 0

## indicator of the last entry being equal to 1
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry_unpenalized)
type lastentry1<:Regularizer
    r::Regularizer
end
prox(r::lastentry1,u::AbstractArray,alpha::Number) = [prox(r.r)(u[1:end-1],alpha), 1]
prox!(r::lastentry1,u::AbstractArray,alpha::Number) = (prox!(r.r)(u[1:end-1],alpha); u[end]=1; u)
evaluate(r::lastentry1,a::AbstractArray) = (a[end]==1 ? evaluate(r.r,a[1:end-1]) : Inf)

## makes the last entry unpenalized
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry1)
type lastentry_unpenalized<:Regularizer
    r::Regularizer
end
prox(r::lastentry_unpenalized,u::AbstractArray,alpha::Number) = [prox(r.r)(u[1:end-1],alpha), u[end]]
prox!(r::lastentry_unpenalized,u::AbstractArray,alpha::Number) = (prox!(r.r)(u[1:end-1],alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray) = evaluate(r.r,a[1:end-1])

## adds an offset to the model by modifying the regularizers
function add_offset(r::Regularizer,rt::Regularizer)
    return lastentry1(r), lastentry_unpenalized(rt)
end

## indicator of 1-sparse vectors
## (enforces that only 1 entry is nonzero, eg for kmeans)
type onesparse<:Regularizer
end
prox(r::onesparse,u::AbstractArray,alpha::Number) = (idx = indmax(u); v=zeros(size(u)); v[idx]=1)
prox!(r::onesparse,u::AbstractArray,alpha::Number) = (idx = indmax(u); scale!(u,0); u[idx]=1)
evaluate(r::onesparse,a::AbstractArray) = sum(map(x->x>0,a)) <= 1 ? 0 : Inf 

# scalings
function equilibrate_variance!(losses::Array{Loss}, A)
    for i=1:size(A,2)
        vari = avgerror(dropna(A[:,i]), losses[i])
        if vari > 0
            losses[i].scale = 1/vari
        else
            losses[i].scale = 0
        end
    end
    return losses
end