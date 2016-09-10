# Predefined regularizers
# You may also implement your own regularizer by subtyping 
# the abstract type Regularizer.
# Regularizers should implement `evaluate` and `prox`. 

import Base.scale!, Roots.fzero

export Regularizer, ProductRegularizer, # abstract types
       # concrete regularizers
       QuadReg, QuadConstraint,
       OneReg, ZeroReg, NonNegConstraint, NonNegOneReg,
       OneSparseConstraint, UnitOneSparseConstraint, SimplexConstraint,
       lastentry1, lastentry_unpenalized, 
       fixed_latent_features, FixedLatentFeaturesConstraint,
       fixed_last_latent_features, FixedLastLatentFeaturesConstraint,
       OrdinalReg,
       # methods on regularizers
       prox!, prox,
       # utilities
       scale, scale!

# numerical tolerance
TOL = 1e-12

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

## QuadLoss regularization
type QuadReg<:Regularizer
    scale::Float64
end
QuadReg() = QuadReg(1)
prox(r::QuadReg,u::AbstractArray,alpha::Number) = 1/(1+2*alpha*r.scale)*u
prox!(r::QuadReg,u::Array{Float64},alpha::Number) = scale!(u, 1/(1+2*alpha*r.scale))
evaluate(r::QuadReg,a::AbstractArray) = r.scale*sum(a.^2)

## constrained QuadLoss regularization
## the function r such that
## r(x) = inf    if norm(x) > max_2norm
##        0      otherwise
## can be used to implement maxnorm regularization: 
##   constraining the maxnorm of XY to be <= mu is achieved 
##   by setting glrm.rx = QuadConstraint(sqrt(mu)) 
##   and the same for every element of glrm.ry
type QuadConstraint<:Regularizer
    max_2norm::Float64
end
QuadConstraint() = QuadConstraint(1)
prox(r::QuadConstraint,u::AbstractArray,alpha::Number) = (r.max_2norm)/norm(u)*u
prox!(r::QuadConstraint,u::Array{Float64},alpha::Number) = scale!(u, (r.max_2norm)/norm(u))
evaluate(r::QuadConstraint,u::AbstractArray) = norm(u) > r.max_2norm + TOL ? Inf : 0
scale(r::QuadConstraint) = 1
scale!(r::QuadConstraint, newscale::Number) = 1

## one norm regularization
type OneReg<:Regularizer
    scale::Float64
end
OneReg() = OneReg(1)
prox(r::OneReg,u::AbstractArray,alpha::Number) = max(u-alpha,0) + min(u+alpha,0)
evaluate(r::OneReg,a::AbstractArray) = r.scale*sum(abs(a))

## no regularization
type ZeroReg<:Regularizer
end
prox(r::ZeroReg,u::AbstractArray,alpha::Number) = u
prox!(r::ZeroReg,u::Array{Float64},alpha::Number) = u
evaluate(r::ZeroReg,a::AbstractArray) = 0
scale(r::ZeroReg) = 0
scale!(r::ZeroReg, newscale::Number) = 0

## indicator of the nonnegative orthant 
## (enforces nonnegativity, eg for nonnegative matrix factorization)
type NonNegConstraint<:Regularizer
end
prox(r::NonNegConstraint,u::AbstractArray,alpha::Number) = broadcast(max,u,0)
prox!(r::NonNegConstraint,u::Array{Float64},alpha::Number) = (@simd for i=1:length(u) @inbounds u[i] = max(u[i], 0) end; u)
function evaluate(r::NonNegConstraint,a::AbstractArray) 
    for ai in a
        if ai<0
            return Inf
        end
    end
    return 0
end
scale(r::NonNegConstraint) = 1
scale!(r::NonNegConstraint, newscale::Number) = 1

## one norm regularization restricted to nonnegative orthant
## (enforces nonnegativity, in addition to one norm regularization)
type NonNegOneReg<:Regularizer
    scale::Float64
end
NonNegOneReg() = NonNegOneReg(1)
prox(r::NonNegOneReg,u::AbstractArray,alpha::Number) = max(u-alpha,0)
function evaluate(r::NonNegOneReg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sum(a)
end
scale(r::NonNegOneReg) = 1
scale!(r::NonNegOneReg, newscale::Number) = 1

## indicator of the last entry being equal to 1
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry_unpenalized)
type lastentry1<:Regularizer
    r::Regularizer
end
lastentry1() = lastentry1(ZeroReg())
prox(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number) = [prox(r.r,view(u,1:length(u)-1),alpha); 1]
prox!(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number) = (prox!(r.r,view(u,1:length(u)-1),alpha); u[end]=1; u)
prox(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number) = [prox(r.r,view(u,1:size(u,1)-1,:),alpha); ones(1, size(u,2))]
prox!(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u[end,:]=1; u)
evaluate(r::lastentry1,a::AbstractArray{Float64,1}) = (a[end]==1 ? evaluate(r.r,a[1:end-1]) : Inf)
evaluate(r::lastentry1,a::AbstractArray{Float64,2}) = (all(a[end,:].==1) ? evaluate(r.r,a[1:end-1,:]) : Inf)
scale(r::lastentry1) = scale(r.r)
scale!(r::lastentry1, newscale::Number) = scale!(r.r, newscale)

## makes the last entry unpenalized
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry1)
type lastentry_unpenalized<:Regularizer
    r::Regularizer
end
lastentry_unpenalized() = lastentry_unpenalized(ZeroReg())
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number) = [prox(r.r,u[1:end-1],alpha); u[end]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number) = (prox!(r.r,view(u,1:size(u,1)-1),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,1}) = evaluate(r.r,a[1:end-1])
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number) = [prox(r.r,u[1:end-1,:],alpha); u[end,:]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,2}) = evaluate(r.r,a[1:end-1,:])
scale(r::lastentry_unpenalized) = scale(r.r)
scale!(r::lastentry_unpenalized, newscale::Number) = scale!(r.r, newscale)

## fixes the values of the first n elements of the column to be y
## optionally regularizes the last k-n elements with regularizer r
type fixed_latent_features<:Regularizer
    r::Regularizer
    y::Array{Float64,1} # the values of the fixed latent features 
    n::Int # length of y
end
fixed_latent_features(r::Regularizer, y::Array{Float64,1}) = fixed_latent_features(r,y,length(y))
# standalone use without another regularizer
FixedLatentFeaturesConstraint(y::Array{Float64, 1}) = fixed_latent_features(ZeroReg(),y,length(y))

prox(r::fixed_latent_features,u::AbstractArray,alpha::Number) = [r.y; prox(r.r,u[(r.n+1):end],alpha)]
function prox!(r::fixed_latent_features,u::Array{Float64},alpha::Number)
  	prox!(r.r,u[(r.n+1):end],alpha)
  	u[1:r.n]=y
  	u
end
evaluate(r::fixed_latent_features, a::AbstractArray) = a[1:r.n]==r.y ? evaluate(r.r, a[(r.n+1):end]) : Inf
scale(r::fixed_latent_features) = scale(r.r)
scale!(r::fixed_latent_features, newscale::Number) = scale!(r.r, newscale)

## fixes the values of the last n elements of the column to be y
## optionally regularizes the first k-n elements with regularizer r
type fixed_last_latent_features<:Regularizer
    r::Regularizer
    y::Array{Float64,1} # the values of the fixed latent features 
    n::Int # length of y
end
fixed_last_latent_features(r::Regularizer, y::Array{Float64,1}) = fixed_last_latent_features(r,y,length(y))
# standalone use without another regularizer
FixedLastLatentFeaturesConstraint(y::Array{Float64, 1}) = fixed_last_latent_features(ZeroReg(),y,length(y))

prox(r::fixed_last_latent_features,u::AbstractArray,alpha::Number) = [prox(r.r,u[(r.n+1):end],alpha); r.y]
function prox!(r::fixed_last_latent_features,u::Array{Float64},alpha::Number)
    u[length(u)-r.n+1:end]=y
    prox!(r.r,u[1:length(a)-r.n],alpha)
    u
end
evaluate(r::fixed_last_latent_features, a::AbstractArray) = a[length(a)-r.n+1:end]==r.y ? evaluate(r.r, a[1:length(a)-r.n]) : Inf
scale(r::fixed_last_latent_features) = scale(r.r)
scale!(r::fixed_last_latent_features, newscale::Number) = scale!(r.r, newscale)

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is nonzero, eg for orthogonal NNMF)
type OneSparseConstraint<:Regularizer
end
prox(r::OneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = indmax(u); v=zeros(size(u)); v[idx]=u[idx]; v)
prox!(r::OneSparseConstraint, u::Array, alpha::Number=0) = (idx = indmax(u); ui = u[idx]; scale!(u,0); u[idx]=ui; u)
function evaluate(r::OneSparseConstraint, a::AbstractArray)
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
scale(r::OneSparseConstraint) = 1
scale!(r::OneSparseConstraint, newscale::Number) = 1

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is 1 and all others are zero, eg for kmeans)
type UnitOneSparseConstraint<:Regularizer
end
prox(r::UnitOneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = indmax(u); v=zeros(size(u)); v[idx]=1; v)
prox!(r::UnitOneSparseConstraint, u::Array, alpha::Number=0) = (idx = indmax(u); scale!(u,0); u[idx]=1; u)
function evaluate(r::UnitOneSparseConstraint, a::AbstractArray)
    oneflag = false
    for ai in a
        if ai==0
            continue
        elseif ai==1
            if oneflag
                return Inf
            else
                oneflag=true
            end
        else
            return Inf
        end
    end
    return 0
end
scale(r::UnitOneSparseConstraint) = 1
scale!(r::UnitOneSparseConstraint, newscale::Number) = 1

## indicator of vectors in the simplex: nonnegative vectors with unit l1 norm
## (eg for QuadLoss mixtures, ie soft kmeans)
## prox for the simplex is derived by Chen and Ye in [this paper](http://arxiv.org/pdf/1101.6081v2.pdf)
type SimplexConstraint<:Regularizer
end
function prox(r::SimplexConstraint, u::AbstractArray, alpha::Number=0)
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
function evaluate(r::SimplexConstraint,a::AbstractArray)
    # check it's a unit vector
    abs(sum(a)-1)>TOL && return Inf
    # check every entry is nonnegative
    for i=1:length(a)
        a[i] < 0 && return Inf
    end
    return 0
end
scale(r::SimplexConstraint) = 1
scale!(r::SimplexConstraint, newscale::Number) = 1

## ordinal regularizer
## a block regularizer which 
    # 1) forces the first k-1 entries of each column to be the same
    # 2) forces the last entry of each column to be increasing
    # 3) applies an internal regularizer to the first k-1 entries of each column
## should always be used in conjunction with lastentry1 regularization on x
type OrdinalReg<:Regularizer
    r::Regularizer
end
OrdinalReg() = OrdinalReg(ZeroReg())
prox(r::OrdinalReg,u::AbstractArray,alpha::Number) = (uc = copy(u); prox!(r,uc,alpha))
function prox!(r::OrdinalReg,u::Array{Float64},alpha::Number)
    um = mean(u[1:end-1, :], 2)
    prox!(r.r,um,alpha)
    for i=1:size(u,1)-1
        for j=1:size(u,2)
            u[i,j] = um[i]
        end
    end
    # this enforces rule 2) (increasing last row of u), but isn't exactly the prox function
    for j=2:size(u,2)
        if u[end,j-1] > u[end,j]
            m = (u[end,j-1] + u[end,j])/2
            u[end,j-1:j] = m
        end
    end
    u
end
evaluate(r::OrdinalReg,a::AbstractArray) = evaluate(r.r,a[1:end-1,1])
scale(r::OrdinalReg) = scale(r.r)
scale!(r::OrdinalReg, newscale::Number) = scale!(r.r, newscale)

# make sure we don't add two offsets cuz that's weird
lastentry_unpenalized(r::OrdinalReg) = r