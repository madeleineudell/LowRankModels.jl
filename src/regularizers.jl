# Predefined regularizers
# You may also implement your own regularizer by subtyping
# the abstract type Regularizer.
# Regularizers should implement `evaluate` and `prox`.

import Base: *

export Regularizer, ProductRegularizer, # abstract types
       # concrete regularizers
       QuadReg, QuadConstraint,
       OneReg, ZeroReg, NonNegConstraint, NonNegOneReg, NonNegQuadReg,
       OneSparseConstraint, UnitOneSparseConstraint, SimplexConstraint,
       KSparseConstraint,
       lastentry1, lastentry_unpenalized,
       fixed_latent_features, FixedLatentFeaturesConstraint,
       fixed_last_latent_features, FixedLastLatentFeaturesConstraint,
       OrdinalReg, MNLOrdinalReg,
       RemQuadReg,
       # methods on regularizers
       prox!, prox,
       # utilities
       scale, mul!, *

# numerical tolerance
TOL = 1e-12

# regularizers
# regularizers r should have the method `prox` defined such that
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)
abstract type Regularizer end
abstract type MatrixRegularizer <: LowRankModels.Regularizer end

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); @simd for i=1:length(u) @inbounds u[i]=v[i] end; u)

# default scaling
scale(r::Regularizer) = r.scale
mul!(r::Regularizer, newscale::Number) = (r.scale = newscale; r)
mul!(rs::Array{Regularizer}, newscale::Number) = (for r in rs mul!(r, newscale) end; rs)
*(newscale::Number, r::Regularizer) = (newr = typeof(r)(); mul!(newr, scale(r)*newscale); newr)

## utilities

function allnonneg(a::AbstractArray)
  for ai in a
    ai < 0 && return false
  end
  return true
end

## QuadLoss regularization
mutable struct QuadReg<:Regularizer
    scale::Float64
end
QuadReg() = QuadReg(1)
prox(r::QuadReg,u::AbstractArray,alpha::Number) = 1/(1+2*alpha*r.scale)*u
prox!(r::QuadReg,u::Array{Float64},alpha::Number) = rmul!(u, 1/(1+2*alpha*r.scale))
evaluate(r::QuadReg,a::AbstractArray) = r.scale*sum(abs2, a)

## constrained QuadLoss regularization
## the function r such that
## r(x) = inf    if norm(x) > max_2norm
##        0      otherwise
## can be used to implement maxnorm regularization:
##   constraining the maxnorm of XY to be <= mu is achieved
##   by setting glrm.rx = QuadConstraint(sqrt(mu))
##   and the same for every element of glrm.ry
mutable struct QuadConstraint<:Regularizer
    max_2norm::Float64
end
QuadConstraint() = QuadConstraint(1)
prox(r::QuadConstraint,u::AbstractArray,alpha::Number) = (r.max_2norm)/norm(u)*u
prox!(r::QuadConstraint,u::Array{Float64},alpha::Number) = mul!(u, (r.max_2norm)/norm(u))
evaluate(r::QuadConstraint,u::AbstractArray) = norm(u) > r.max_2norm + TOL ? Inf : 0
scale(r::QuadConstraint) = 1
mul!(r::QuadConstraint, newscale::Number) = 1

## one norm regularization
mutable struct OneReg<:Regularizer
    scale::Float64
end
OneReg() = OneReg(1)
prox(r::OneReg,u::AbstractArray,alpha::Number) = max.(u-alpha,0) + min.(u+alpha,0)
prox!(r::OneReg,u::AbstractArray,alpha::Number) = begin
  softthreshold = (x::Number -> max.(x-alpha,0) + min.(x+alpha,0))
  map!(softthreshold, u, u)
end
evaluate(r::OneReg,a::AbstractArray) = r.scale*sum(abs,a)


## no regularization
mutable struct ZeroReg<:Regularizer
end
prox(r::ZeroReg,u::AbstractArray,alpha::Number) = u
prox!(r::ZeroReg,u::Array{Float64},alpha::Number) = u
evaluate(r::ZeroReg,a::AbstractArray) = 0
scale(r::ZeroReg) = 0
mul!(r::ZeroReg, newscale::Number) = 0

## indicator of the nonnegative orthant
## (enforces nonnegativity, eg for nonnegative matrix factorization)
mutable struct NonNegConstraint<:Regularizer
end
prox(r::NonNegConstraint,u::AbstractArray,alpha::Number=1) = broadcast(max,u,0)
prox!(r::NonNegConstraint,u::Array{Float64},alpha::Number=1) = (@simd for i=1:length(u) @inbounds u[i] = max(u[i], 0) end; u)
function evaluate(r::NonNegConstraint,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return 0
end
scale(r::NonNegConstraint) = 1
mul!(r::NonNegConstraint, newscale::Number) = 1

## one norm regularization restricted to nonnegative orthant
## (enforces nonnegativity, in addition to one norm regularization)
mutable struct NonNegOneReg<:Regularizer
    scale::Float64
end
NonNegOneReg() = NonNegOneReg(1)
prox(r::NonNegOneReg,u::AbstractArray,alpha::Number) = max.(u-alpha,0)

prox!(r::NonNegOneReg,u::AbstractArray,alpha::Number) = begin
  nonnegsoftthreshold = (x::Number -> max.(x-alpha,0))
  map!(nonnegsoftthreshold, u)
end

function evaluate(r::NonNegOneReg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sum(a)
end
scale(r::NonNegOneReg) = 1
mul!(r::NonNegOneReg, newscale::Number) = 1

## Quadratic regularization restricted to nonnegative domain
## (Enforces nonnegativity alongside quadratic regularization)
mutable struct NonNegQuadReg
    scale::Float64
end
NonNegQuadReg() = NonNegQuadReg(1)
prox(r::NonNegQuadReg,u::AbstractArray,alpha::Number) = max.(1/(1+2*alpha*r.scale)*u, 0)
prox!(r::NonNegQuadReg,u::AbstractArray,alpha::Number) = begin
  mul!(u, 1/(1+2*alpha*r.scale))
  maxval = maximum(u)
  clamp!(u, 0, maxval)
end
function evaluate(r::NonNegQuadReg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sumabs2(a)
end

## indicator of the last entry being equal to 1
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry_unpenalized)
mutable struct lastentry1<:Regularizer
    r::Regularizer
end
lastentry1() = lastentry1(ZeroReg())
prox(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number=1) = [prox(r.r,view(u,1:length(u)-1),alpha); 1]
prox!(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number=1) = (prox!(r.r,view(u,1:length(u)-1),alpha); u[end]=1; u)
prox(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number=1) = [prox(r.r,view(u,1:size(u,1)-1,:),alpha); ones(1, size(u,2))]
prox!(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u[end,:]=1; u)
evaluate(r::lastentry1,a::AbstractArray{Float64,1}) = (a[end]==1 ? evaluate(r.r,a[1:end-1]) : Inf)
evaluate(r::lastentry1,a::AbstractArray{Float64,2}) = (all(a[end,:].==1) ? evaluate(r.r,a[1:end-1,:]) : Inf)
scale(r::lastentry1) = scale(r.r)
mul!(r::lastentry1, newscale::Number) = mul!(r.r, newscale)

## makes the last entry unpenalized
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry1)
mutable struct lastentry_unpenalized<:Regularizer
    r::Regularizer
end
lastentry_unpenalized() = lastentry_unpenalized(ZeroReg())
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number=1) = [prox(r.r,u[1:end-1],alpha); u[end]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,1}) = evaluate(r.r,a[1:end-1])
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number=1) = [prox(r.r,u[1:end-1,:],alpha); u[end,:]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,2}) = evaluate(r.r,a[1:end-1,:])
scale(r::lastentry_unpenalized) = scale(r.r)
mul!(r::lastentry_unpenalized, newscale::Number) = mul!(r.r, newscale)

## fixes the values of the first n elements of the column to be y
## optionally regularizes the last k-n elements with regularizer r
mutable struct fixed_latent_features<:Regularizer
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
mul!(r::fixed_latent_features, newscale::Number) = mul!(r.r, newscale)

## fixes the values of the last n elements of the column to be y
## optionally regularizes the first k-n elements with regularizer r
mutable struct fixed_last_latent_features<:Regularizer
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
mul!(r::fixed_last_latent_features, newscale::Number) = mul!(r.r, newscale)

## indicator of 1-sparse vectors
## (enforces that exact 1 entry is nonzero, eg for orthogonal NNMF)
mutable struct OneSparseConstraint<:Regularizer
end
prox(r::OneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = argmax(u); v=zeros(size(u)); v[idx]=u[idx]; v)
prox!(r::OneSparseConstraint, u::Array, alpha::Number=0) = (idx = argmax(u); ui = u[idx]; mul!(u,0); u[idx]=ui; u)
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
mul!(r::OneSparseConstraint, newscale::Number) = 1

## Indicator of k-sparse vectors
mutable struct KSparseConstraint<:Regularizer
  k::Int
end
function evaluate(r::KSparseConstraint, a::AbstractArray)
  k = r.k
  nonzcount = 0
  for ai in a
    if nonzcount == k
      if ai != 0
        return Inf
      end
    else
      if ai != 0
        nonzcount += 1
      end
    end
  end
  return 0
end
function prox(r::KSparseConstraint, u::AbstractArray, alpha::Number)
  k = r.k
  ids = partialsortperm(u, 1:k, by=abs, rev=true)
  uk = zero(u)
  uk[ids] = u[ids]
  uk
end
function prox!(r::KSparseConstraint, u::Array, alpha::Number)
  k = r.k
  ids = partialsortperm(u, 1:k, by=abs, rev=true)
  vals = u[ids]
  mul!(u,0)
  u[ids] = vals
  u
end

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is 1 and all others are zero, eg for kmeans)
mutable struct UnitOneSparseConstraint<:Regularizer
end
prox(r::UnitOneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = argmax(u); v=zeros(size(u)); v[idx]=1; v)
prox!(r::UnitOneSparseConstraint, u::Array, alpha::Number=0) = (idx = argmax(u); mul!(u,0); u[idx]=1; u)

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
mul!(r::UnitOneSparseConstraint, newscale::Number) = 1

## indicator of vectors in the simplex: nonnegative vectors with unit l1 norm
## (eg for QuadLoss mixtures, ie soft kmeans)
## prox for the simplex is derived by Chen and Ye in [this paper](http://arxiv.org/pdf/1101.6081v2.pdf)
mutable struct SimplexConstraint<:Regularizer
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
    max.(u .- t, 0)
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
mul!(r::SimplexConstraint, newscale::Number) = 1

## ordinal regularizer
## a block regularizer which
    # 1) forces the first k-1 entries of each column to be the same
    # 2) forces the last entry of each column to be increasing
    # 3) applies an internal regularizer to the first k-1 entries of each column
## should always be used in conjunction with lastentry1 regularization on x
mutable struct OrdinalReg<:Regularizer
    r::Regularizer
end
OrdinalReg() = OrdinalReg(ZeroReg())
prox(r::OrdinalReg,u::AbstractArray,alpha::Number) = (uc = copy(u); prox!(r,uc,alpha))
function prox!(r::OrdinalReg,u::AbstractArray,alpha::Number)
    um = mean(u[1:end-1, :], dims=2)
    prox!(r.r,um,alpha)
    for i=1:size(u,1)-1
        for j=1:size(u,2)
            u[i,j] = um[i]
        end
    end
    # this enforces rule 2) (increasing last row of u), but isn't exactly the prox function
    # for j=2:size(u,2)
    #     if u[end,j-1] > u[end,j]
    #         m = (u[end,j-1] + u[end,j])/2
    #         u[end,j-1:j] = m
    #     end
    # end
    u
end
evaluate(r::OrdinalReg,a::AbstractArray) = evaluate(r.r,a[1:end-1,1])
scale(r::OrdinalReg) = scale(r.r)
mul!(r::OrdinalReg, newscale::Number) = mul!(r.r, newscale)

# make sure we don't add two offsets cuz that's weird
lastentry_unpenalized(r::OrdinalReg) = r

mutable struct MNLOrdinalReg<:Regularizer
    r::Regularizer
end
MNLOrdinalReg() = MNLOrdinalReg(ZeroReg())
prox(r::MNLOrdinalReg,u::AbstractArray,alpha::Number) = (uc = copy(u); prox!(r,uc,alpha))
function prox!(r::MNLOrdinalReg,u::AbstractArray,alpha::Number; TOL=1e-3)
    um = mean(u[1:end-1, :], dims=2)
    prox!(r.r,um,alpha)
    for i=1:size(u,1)-1
        for j=1:size(u,2)
            u[i,j] = um[i]
        end
    end
    # this enforces rule 2) (decreasing last row of u, all less than 0), but isn't exactly the prox function
    u[end,1] = min(-TOL, u[end,1])
    for j=2:size(u,2)
      u[end,j] = min(u[end,j], u[end,j-1]-TOL)
    end
    u
end
evaluate(r::MNLOrdinalReg,a::AbstractArray) = evaluate(r.r,a[1:end-1,1])
scale(r::MNLOrdinalReg) = scale(r.r)
mul!(r::MNLOrdinalReg, newscale::Number) = mul!(r.r, newscale)
# make sure we don't add two offsets cuz that's weird
lastentry_unpenalized(r::MNLOrdinalReg) = r

## Quadratic regularization with non-zero mean
mutable struct RemQuadReg<:Regularizer
        scale::Float64
        m::Array{Float64, 1}
end
RemQuadReg(m::Array{Float64, 1}) = RemQuadReg(1, m)
prox(r::RemQuadReg, u::AbstractArray, alpha::Number) =
     (u + 2 * alpha * r.scale * r.m) / (1 + 2 * alpha * r.scale)
prox!(r::RemQuadReg, u::Array{Float64}, alpha::Number) = begin
        broadcast!(.+, u, u, 2 * alpha * r.scale * r.m)
        mul!(u, 1 / (1 + 2 * alpha * r.scale))
end
evaluate(r::RemQuadReg, a::AbstractArray) = r.scale * sum(abs2, a - r.m)

## simpler method for numbers, not arrays
evaluate(r::Regularizer, u::Number) = evaluate(r, [u])
prox(r::Regularizer, u::Number, alpha::Number) = prox(r, [u], alpha)[1]
# if step size not specified, step size = 1
prox(r::Regularizer, u) = prox(r, u, 1)
