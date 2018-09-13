### Streaming method
#   only implemented for quadratic objectives
#   TODO: add quadratic regularization

export StreamingParams, streaming_fit!, streaming_impute!

type StreamingParams<:AbstractParams
    T0::Int # number of rows to use to initialize Y before streaming begins
    stepsize::Float64 # stepsize (inverse of memory)
    Y_update_interval::Int # how often to prox Y
end
function StreamingParams(
    T0::Int=1000; # number of rows to use to initialize Y before streaming begins
    stepsize::Number=1/T0, # (inverse of memory)
    Y_update_interval::Int=10 # how often to prox Y

)
    return StreamingParams(T0, convert(Float64, stepsize), Y_update_interval)
end

### FITTING
function streaming_fit!(glrm::GLRM, params::StreamingParams=StreamingParams();
    ch::ConvergenceHistory=ConvergenceHistory("StreamingGLRM"),
    verbose=true)

  # make sure everything is quadratic
  @assert all(map(l->isa(l, QuadLoss), glrm.losses))
  @assert all(map(l->isa(l, QuadReg), glrm.rx))
  @assert all(map(l->isa(l, QuadReg), glrm.ry))

  # initialize Y and first T0 rows of X
  init_glrm = keep_rows(glrm, params.T0)
  init_svd!(init_glrm)
  copy!(glrm.Y, init_glrm.Y)
  copy!(view(glrm.X, :, 1:params.T0), init_glrm.X)

  ### initialization
  A = glrm.A # rename these for easier local access
  rx = glrm.rx
  ry = glrm.ry
  X = glrm.X; Y = glrm.Y
  k = glrm.k
  m,n = size(A)

  # yscales = map(r->r.scale, ry)

  for i=params.T0+1:m
    # update x_i
    obs = glrm.observed_features[i]
    Yobs = Y[:, obs]
    Aobs = A[i, obs]
    xi = view(X, :, i)

    copy!(xi, (Yobs * Yobs' + 2 * rx[i].scale * I) \ (Yobs * Aobs))

    # update objective
    r = Yobs'*xi - Aobs
    push!(ch.objective, norm(r) ^ 2)

    # # update Y
    # TODO verify this is stochastic proximal gradient (with constant stepsize) for the problem
    # TODO don't prox Y at every iteration
    # TODO don't assume scales on all the rys are equal
    # gY[:, jj] = xi * r' == r[jj] * xi # gradient of ith row objective wrt Y
    for jj in 1:length(obs)
      Y[:,obs[jj]] -= params.stepsize * r[jj] * xi
    end
    if i%params.Y_update_interval == 0
      # prox!(ry, Y, params.stepsize * params.Y_update_interval)
      Y ./= (1 + 2 * params.stepsize * params.Y_update_interval * ry[1].scale)
    end
  end

  return X, Y, ch
end

### FITTING
function streaming_impute!(glrm::GLRM, params::StreamingParams=StreamingParams();
    ch::ConvergenceHistory=ConvergenceHistory("StreamingGLRM"),
    verbose=true)

  # make sure everything is quadratic
  @assert all(map(l->isa(l, QuadLoss), glrm.losses))
  @assert all(map(l->isa(l, QuadReg), glrm.rx))
  @assert all(map(l->isa(l, QuadReg), glrm.ry))

  # initialize Y and first T0 rows of X
  init_glrm = keep_rows(glrm, params.T0)
  init_svd!(init_glrm)
  copy!(glrm.Y, init_glrm.Y)
  copy!(view(glrm.X, :, 1:params.T0), init_glrm.X)

  ### initialization
  A = glrm.A # rename these for easier local access
  Ahat = copy(glrm.A)
  rx = glrm.rx
  ry = glrm.ry
  X = glrm.X; Y = glrm.Y
  k = glrm.k
  m,n = size(A)

  # yscales = map(r->r.scale, ry)

  for i=params.T0+1:m
    # update x_i
    obs = glrm.observed_features[i]
    Yobs = Y[:, obs]
    Aobs = A[i, obs]
    xi = view(X, :, i)

    copy!(xi, (Yobs * Yobs' + 2 * rx[i].scale * I) \ (Yobs * Aobs))

    # impute
    not_obs = setdiff(Set(1:n), Set(obs))
    if length(not_obs)>0
      ahat = xi'*Y
      Ahat[i, not_obs] = ahat[not_obs]
    end

    # update objective
    r = Yobs'*xi - Aobs
    push!(ch.objective, norm(r) ^ 2)

    # # update Y
    # TODO verify this is stochastic proximal gradient (with constant stepsize) for the problem
    # TODO don't prox Y at every iteration
    # TODO don't assume scales on all the rys are equal
    # gY[:, jj] = xi * r' == r[jj] * xi # gradient of ith row objective wrt Y
    for jj in 1:length(obs)
      Y[:,obs[jj]] -= params.stepsize * r[jj] * xi
    end
    if i%params.Y_update_interval == 0
      # prox!(ry, Y, params.stepsize * params.Y_update_interval)
      Y ./= (1 + 2 * params.stepsize * params.Y_update_interval * ry[1].scale)
    end
  end

  return Ahat
end

""" Constructs new GLRM on subset of rows of the data from input glrm """
function keep_rows(glrm, r::Range{Int})
  @assert maximum(r) <= size(glrm.A, 1)
  obs = flatten_observations(glrm.observed_features)
  first_row = minimum(r)
  if first_row > 1
    new_obs = map( t -> (t[1]-first_row+1, t[2]), filter( t -> (t[1] in r), obs))
  else
    new_obs = filter( t -> (t[1] in r), obs)
  end
  of, oe = sort_observations(new_obs, length(r), size(glrm.A, 2))
  new_glrm = GLRM(glrm.A[r,:], glrm.losses, glrm.rx[r], glrm.ry, glrm.k,
                  observed_features = of, observed_examples = oe)
  return new_glrm
end
keep_rows(glrm, T::Int) = keep_rows(glrm, 1:T)
