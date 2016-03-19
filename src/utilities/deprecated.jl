using Base.depwarn

@compat Base.@deprecate GLRM(A::AbstractArray, obs::Array{Tuple{Int, Int}, 1}, args...; kwargs...) GLRM(A, args...; obs = obs, kwargs...)

Base.@deprecate ProxGradParams(s::Number,m::Int,c::Float64,ms::Float64) ProxGradParams(s, max_iter=m, abs_tol=c, min_stepsize=ms)

Base.@deprecate expand_categoricals expand_categoricals!

Base.@deprecate errors(g::GLRM) error_metric(g)

Base.@deprecate quadratic QuadLoss

Base.@deprecate logistic LogisticLoss

Base.@deprecate huber HuberLoss

Base.@deprecate LogLoss LogisticLoss

Base.@deprecate l1 L1Loss

Base.@deprecate poisson PoissonLoss

Base.@deprecate ordinal_hinge OrdinalHingeLoss

Base.@deprecate OrdinalHinge OrdinalHingeLoss

Base.@deprecate WeightedHinge WeightedHingeLoss

Base.@deprecate periodic PeriodicLoss

Base.@deprecate quadreg QuadReg

Base.@deprecate constrained_quadreg QuadConstraint

Base.@deprecate onereg OneReg

Base.@deprecate zeroreg ZeroReg

Base.@deprecate nonnegative NonNegConstraint

Base.@deprecate onesparse OneSparseConstraint

Base.@deprecate unitonesparse UnitOneSparseConstraint

Base.@deprecate simplex SimplexConstraint

Base.@deprecate nonneg_onereg NonNegOneReg
