export sort_observations, add_offset!, fix_latent_features!,
       equilibrate_variance!, prob_scale!

### OBSERVATION TUPLES TO ARRAYS
function sort_observations(obs::Array{Tuple{Int,Int},1}, m::Int, n::Int; check_empty=false)
    observed_features = Array{Int,1}[Int[] for i=1:m]
    observed_examples = Array{Int,1}[Int[] for j=1:n]
    for (i,j) in obs
        push!(observed_features[i],j)
        push!(observed_examples[j],i)
    end
    if check_empty && (any(map(x->length(x)==0,observed_examples)) ||
            any(map(x->length(x)==0,observed_features)))
        error("Every row and column must contain at least one observation")
    end
    return observed_features, observed_examples
end

### SCALINGS AND OFFSETS ON GLRM
function add_offset!(glrm::AbstractGLRM)
    glrm.rx, glrm.ry = map(lastentry1, glrm.rx), map(lastentry_unpenalized, glrm.ry)
    return glrm
end
function fix_latent_features!(glrm::AbstractGLRM, n)
    glrm.ry = Regularizer[fixed_latent_features(glrm.ry[i], glrm.Y[1:n,i])
                            for i in 1:length(glrm.ry)]
    return glrm
end

## equilibrate variance
# scale all columns inversely proportional to mean value of loss function
# makes sense when all loss functions used are nonnegative
function equilibrate_variance!(glrm::AbstractGLRM, columns_to_scale = 1:size(glrm.A,2))
    for i in columns_to_scale
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
            mul!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
        end
        if varregi > 0
            mul!(glrm.ry[i], scale(glrm.ry[i])/varregi)
        end
    end
    return glrm
end

## probabilistic scaling
# scale loss function to fit -loglik of joint distribution
# makes sense when all functions used are -logliks of sensible distributions
# todo: option to scale to account for nonuniform sampling in rows or columns or both
# skipmissing(Array with missing) gives an iterator.
function prob_scale!(glrm, columns_to_scale = 1:size(glrm.A,2))
    for i in columns_to_scale
        nomissing = glrm.A[glrm.observed_examples[i],i]
        if typeof(glrm.losses[i]) == QuadLoss && length(nomissing) > 0
            varlossi = var(skipmissing(glrm.A[i])) # estimate the variance
            if varlossi > TOL
    		  mul!(glrm.losses[i], 1/(2*varlossi)) # this is the correct -loglik of gaussian with variance fixed at estimate
    	    else
    		  warn("column $i has a variance of $varlossi; not scaling it to avoid dividing by zero.")
    	    end
        elseif typeof(glrm.losses[i]) == HuberLoss && length(nomissing) > 0
            varlossi = avgerror(glrm.losses[i], glrm.A[i]) # estimate the width of the distribution
            if varlossi > TOL
              mul!(glrm.losses[i], 1/(2*varlossi)) # this is not the correct -loglik of huber with estimates for variance and mean of poisson, but that's probably ok
            else
              warn("column $i has a variance of $varlossi; not scaling it to avoid dividing by zero.")
            end
        else # none of the other distributions have any free parameters to estimate, so this is the correct -loglik
            mul!(glrm.losses[i], 1)
        end
    end
    return glrm
end
