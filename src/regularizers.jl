# Predefined regularizers
# You may also implement your own regularizer by subtyping 
# the abstract type Regularizer.
# Regularizers should implement `evaluate` and `prox`. 

# TO DO:
# document this stuff better
# tidy up the interfaces a la losses.jl

import Base.scale!, Roots.fzero

export Regularizer, # abstract type
       # concrete regularizers
       quadreg, constrained_quadreg,
       onereg, zeroreg, nonnegative, nonneg_onereg,
       onesparse, unitonesparse, simplex,
       lastentry1, lastentry_unpenalized, fixed_latent_features,
       # methods on regularizers
       prox!, prox,
       # utilities
       scale, scale!

# regularizers
# regularizers r should have the method `prox` defined such that 
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)
abstract Regularizer

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); @simd for i=1:length(u) @inbounds u[i]=v[i] end; u)
scale(r::Regularizer) = r.scale
scale!(r::Regularizer, newscale::Number) = (r.scale = newscale; r)
scale!(rs::Array{Regularizer}, newscale::Number) = (for r in rs scale!(r, newscale) end; rs)

## utilities

function allnonneg(a::AbstractArray)
  for ai in a
    ai < 0 && return false
  end
  return true
end

## quadratic regularization
type quadreg<:Regularizer
    scale::Float64
end
quadreg() = quadreg(1)
prox(r::quadreg,u::AbstractArray,alpha::Number) = 1/(1+alpha*r.scale/2)*u
prox!(r::quadreg,u::Array{Float64},alpha::Number) = scale!(u, 1/(1+alpha*r.scale/2))
evaluate(r::quadreg,a::AbstractArray) = r.scale*sum(a.^2)

## constrained quadratic regularization
## the function r such that
## r(x) = inf    if norm(x) > max_2norm
##        0      otherwise
## can be used to implement maxnorm regularization: 
##   constraining the maxnorm of XY to be <= mu is achieved 
##   by setting glrm.rx = constrained_quadreg(sqrt(mu)) 
##   and the same for every element of glrm.ry
type constrained_quadreg<:Regularizer
    max_2norm::Float64
end
constrained_quadreg() = constrained_quadreg(1)
prox(r::constrained_quadreg,u::AbstractArray,alpha::Number) = (r.max_2norm)/norm(u)*u
prox!(r::constrained_quadreg,u::Array{Float64},alpha::Number) = scale!(u, (r.max_2norm)/norm(u))
evaluate(r::constrained_quadreg,u::AbstractArray) = norm(u) > r.max_2norm ? Inf : 0
scale(r::constrained_quadreg) = 1
scale!(r::constrained_quadreg, newscale::Number) = 1

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
function evaluate(r::nonnegative,a::AbstractArray) 
    for ai in a
        if ai<0
            return Inf
        end
    end
    return 0
end
scale(r::nonnegative) = 1
scale!(r::nonnegative, newscale::Number) = 1

## one norm regularization restricted to nonnegative orthant
## (enforces nonnegativity, in addition to one norm regularization)
type nonneg_onereg<:Regularizer
    scale::Float64
end
nonneg_onereg() = nonneg_onereg(1)
prox(r::nonneg_onereg,u::AbstractArray,alpha::Number) = max(u-alpha,0)
function evaluate(r::nonneg_onereg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sum(a)
end
scale(r::nonneg_onereg) = 1
scale!(r::nonneg_onereg, newscale::Number) = 1

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

type fixed_latent_features<:Regularizer
    r::Regularizer
    y::Array{Float64,1} # the values of the fixed latent features 
    n::Int # length of y
end
fixed_latent_features(r::Regularizer, y::Array{Float64,1}) = fixed_latent_features(r,y,length(y))
prox(r::fixed_latent_features,u::AbstractArray,alpha::Number) = [r.y, prox(r.r,u[(r.n+1):end],alpha)]
function prox!(r::fixed_latent_features,u::Array{Float64},alpha::Number)
  	prox!(r.r,u[(r.n+1):end],alpha)
  	u[1:r.n]=y
  	u
end
evaluate(r::fixed_latent_features, a::AbstractArray) = a[1:r.n]==r.y ? evaluate(r.r, a[(r.n+1):end]) : Inf
scale(r::fixed_latent_features) = r.r.scale
scale!(r::fixed_latent_features, newscale::Number) = (r.r.scale = newscale)

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is nonzero, eg for orthogonal NNMF)
type onesparse<:Regularizer
end
prox(r::onesparse, u::AbstractArray, alpha::Number) = (idx = indmax(u); v=zeros(size(u)); v[idx]=u[idx]; v)
prox!(r::onesparse, u::Array, alpha::Number) = (idx = indmax(u); ui = u[idx]; scale!(u,0); u[idx]=ui; u)
function evaluate(r::onesparse, a::AbstractArray)
    oneflag = false
    for ai in a
        if oneflag
            if ai!=0
                return Inf
            end
        else
            if ai!=0
                oneflag=true
            end
        end
    end
    return 0
end
scale(r::onesparse) = 1
scale!(r::onesparse, newscale::Number) = 1

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is 1 and all others are zero, eg for kmeans)
type unitonesparse<:Regularizer
end
prox(r::unitonesparse, u::AbstractArray, alpha::Number) = (idx = indmax(u); v=zeros(size(u)); v[idx]=1; v)
prox!(r::unitonesparse, u::Array, alpha::Number) = (idx = indmax(u); scale!(u,0); u[idx]=1; u)
function evaluate(r::unitonesparse, a::AbstractArray)
    oneflag = false
    for ai in a
        if oneflag
            if ai==1
                return Inf
            end
        else
            if ai==1
                oneflag=true
            end
        end
    end
    return 0
end
scale(r::unitonesparse) = 1
scale!(r::unitonesparse, newscale::Number) = 1

## indicator of vectors in the simplex: nonnegative vectors with unit l1 norm
## (eg for quadratic mixtures, ie soft kmeans)
## prox for the simplex is derived by Chen and Ye in [this paper](http://arxiv.org/pdf/1101.6081v2.pdf)
type simplex<:Regularizer
end
function prox(r::simplex, u::AbstractArray, alpha::Number)
    n = length(u)
    y = sort(u, rev=true)
    ysum = cumsum(y)
    t = (ysum[end]-1)/n
    for i=1:(n-1)
        if (ysum[i]-1)/i >= y[i+1]
            t = (ysum[i]-1)/i
            break
        end
    end
    max(u - t, 0)
end
function evaluate(r::simplex, a::AbstractArray)
    for ai in a
        if ai>=1 || ai<=0
            return Inf
        end
    end
    if sum(a) != 1
        return Inf
    end
    return 0
end
scale(r::simplex) = 1
scale!(r::simplex, newscale::Number) = 1

