##############################################################
### copying
##############################################################

import Base.copy
export copy, GLRM

function copy(r::Regularizer)
  newr = typeof(r)()
  for field in @compat fieldnames(r)
    setfield!(newr, field, copy(getfield(r, field)))
  end
  newr
end
function copy(r::Loss)
  newr = typeof(r)()
  for field in @compat fieldnames(r)
    setfield!(newr, field, copy(getfield(r, field)))
  end
  newr
end
function copy(r::AbstractGLRM)
  newr = typeof(r)()
  for field in @compat fieldnames(r)
    setfield!(newr, field, copy(getfield(r, field)))
  end
  newr
end
# points to all the same problem data as the original input GLRM, 
# but copies the estimate of the model parameters
function copy_estimate(g::GLRM)
  return GLRM(g.A,g.losses,g.rx,g.ry,g.k,
              g.observed_features,g.observed_examples,
              copy(g.X),copy(g.Y))
end
function copy_estimate(g::GFRM)
  return GFRM(g.A,g.losses,g.r,g.k,
              g.observed_features,g.observed_examples,
              copy(g.U),copy(g.W))
end
# function copy_estimate(r::GLRM)
#   newr = typeof(r)()
#   for field in @compat fieldnames(r)
#     setfield!(newr, field, getfield(r, field))
#   end
#   newr.X = copy(r.X)
#   newr.Y = copy(r.Y)
#   newr
# end
# function copy_estimate(r::GFRM)
#   newr = typeof(r)()
#   for field in @compat fieldnames(r)
#     setfield!(newr, field, getfield(r, field))
#   end
#   newr.U = copy(r.U)
#   newr.W = copy(r.W)
#   newr
# end


# domains are immutable, so this is ok
copy(d::Domain) = d

##############################################################
### fill singleton losses and regularizers to the right shapes
##############################################################

# fill an array of length n with copies of the object foo
fillcopies(foo, n::Int; arraytype=typeof(foo)) = arraytype[copy(foo) for i=1:n]

# singleton loss:
GLRM(A, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int; kwargs...) = 
    GLRM(A, fillcopies(loss, size(A, 2)), rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), k; kwargs...)
GLRM(A, loss::Loss, rx::Regularizer, ry::Array, k::Int; kwargs...) = 
    GLRM(A, fillcopies(loss, size(A, 2), arraytype=Loss), rx, ry, k; kwargs...)

# singleton regularizer on y:
GLRM(A, losses::Array, rx::Regularizer, ry::Regularizer, k::Int; kwargs...) = 
    GLRM(A, losses, rx, fillcopies(ry, size(A, 2), arraytype=Regularizer), k::Int; kwargs...)
