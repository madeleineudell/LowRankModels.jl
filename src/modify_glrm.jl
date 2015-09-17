export sort_observations, add_offset!, equilibrate_variance!, fix_latent_features!

### OBSERVATION TUPLES TO ARRAYS
@compat function sort_observations(obs::Array{Tuple{Int,Int},1}, m::Int, n::Int; check_empty=false)
    observed_features = Array{Int,1}[Int[] for i=1:m]
    observed_examples = Array{Int,1}[Int[] for j=1:n]
    for (i,j) in obs
        @inbounds push!(observed_features[i],j)
        @inbounds push!(observed_examples[j],i)
    end
    if check_empty && (any(map(x->length(x)==0,observed_examples)) || 
            any(map(x->length(x)==0,observed_features)))
        error("Every row and column must contain at least one observation")
    end
    return observed_features, observed_examples
end

## SCALINGS AND OFFSETS ON GLRM
function add_offset!(glrm::AbstractGLRM)
    glrm.rx, glrm.ry = lastentry1(glrm.rx), map(lastentry_unpenalized, glrm.ry)
    return glrm
end
function equilibrate_variance!(glrm::AbstractGLRM)
    for i=1:size(glrm.A,2)
        nomissing = glrm.A[glrm.observed_examples[i],i]
        if length(nomissing)>0
            varlossi = avgerror(glrm.losses[i], nomissing)
            varregi = var(nomissing) # TODO make this depend on the kind of regularization; this assumes QuadLoss
        else
            varlossi = 1
            varregi = 1
        end
        if varlossi > 0
            # rescale the losses and regularizers for each column by the inverse of the empirical variance
            scale!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
        end
        if varregi > 0
            scale!(glrm.ry[i], scale(glrm.ry[i])/varregi)
        end
    end
    return glrm
end
function fix_latent_features!(glrm::AbstractGLRM, n)
    glrm.ry = Regularizer[fixed_latent_features(glrm.ry[i], glrm.Y[1:n,i]) 
                            for i in 1:length(glrm.ry)]
    return glrm
end