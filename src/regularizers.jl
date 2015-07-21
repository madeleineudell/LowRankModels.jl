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
       quadreg, onereg, zeroreg, nonnegative, nonneg_onereg,
       onesparse, unitonesparse, simplex, poisson_sparse,
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

## sum regularization for poisson errors
type poisson_sparse<:Regularizer
    scale::Float64
end
poisson_sparse() = poisson_sparse(1)
function prox(r::poisson_sparse,u::AbstractArray,alpha::Number) 
    uprox = zeros(size(u))
    for i in 1:length(u)
        uprox[i] = fzero(x->x+r.scale*alpha*exp(x)-u[i], -50, 50)
    end
    return uprox
end
evaluate(r::poisson_sparse,a::AbstractArray) = r.scale*sum(exp(a))

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

## one norm regularization restricted to nonnegative orthant
## (enforces nonnegativity, in addition to one norm regularization)
type nonneg_onereg<:Regularizer
    scale::Float64
end
nonneg_onereg() = nonneg_onereg(1)
prox(r::nonneg_onereg,u::AbstractArray,alpha::Number) = max(u-alpha,0)
evaluate(r::nonneg_onereg,a::AbstractArray) = any(map(x->x<0,a)) ? Inf : r.scale*sum(a)

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
evaluate(r::fixed_latent_features,a::AbstractArray) = a[1:r.n]==r.y ? evaluate(r.r, a[(r.n+1):end]) : Inf
scale(r::fixed_latent_features) = r.r.scale
scale!(r::fixed_latent_features, newscale::Number) = (r.r.scale = newscale)

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

## indicator of vectors in the simplex: nonnegative vectors with unit l1 norm
## (eg for quadratic mixtures, ie soft kmeans)
## prox for the simplex is derived by Chen and Ye in [this paper](http://arxiv.org/pdf/1101.6081v2.pdf)
type simplex<:Regularizer
end
function prox!(r::simplex,u::AbstractArray,alpha::Number)
    n = length(u)
    y = sort(u, rev=true)
    ysum = cumsum(y)
    t = ysum[end]/n
    for i=1:n-1
        if (ysum[i] - 1)/i >= y[i+1]
            t = (ysum[i] - 1)/i
            break
        end
    end
    u = max(u - t, 0)
end
evaluate(r::simplex,a::AbstractArray) = ((sum(map(x->x>=0.0,a)) <= 1.0 && sum(a)==1) ? 0.0 : Inf )
scale(r::simplex) = 1
scale!(r::simplex, newscale::Number) = 1