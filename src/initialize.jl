import StatsBase: sample, wsample
export init_kmeanspp!, init_svd!, init_nndsvd!
import Arpack: svds

# kmeans++ initialization, but with missing data
# we make sure never to look at "unobserved" entries in A
# so that models can be honestly cross validated, for example
function init_kmeanspp!(glrm::GLRM)
	m,n = size(glrm.A)
	k = glrm.k
	possible_centers = Set(1:m)
	glrm.Y = randn(k,n)
	# assign first center randomly
	i = sample(1:m)
	setdiff!(possible_centers, i)
	glrm.Y[1,glrm.observed_features[i]] = glrm.A[i,glrm.observed_features[i]]
	# assign next centers one by one
	for l=1:k-1
		min_dists_per_obs = zeros(m)
		for i in possible_centers
			d = zeros(l)
			for j in glrm.observed_features[i]
				for ll=1:l
					d[ll] += evaluate(glrm.losses[j], glrm.Y[ll,j], glrm.A[i,j])
				end
			end
			min_dists_per_obs[i] = minimum(d)/length(glrm.observed_features[i])
		end
        furthest_index = wsample(1:m,min_dists_per_obs)
		glrm.Y[l+1,glrm.observed_features[furthest_index]] = glrm.A[furthest_index,glrm.observed_features[furthest_index]]
	end
	return glrm
end

function init_svd!(glrm::GLRM; offset=true, scale=true, TOL = 1e-10)
    # only offset if the glrm model is offset
    offset = offset && typeof(glrm.rx) == lastentry1
    # only scale if we also offset
    scale = scale && offset
    m,n = size(glrm.A)
    k = glrm.k

    # find spans of loss functions (for multidimensional losses)
    yidxs = get_yidxs(glrm.losses)
    d = maximum(yidxs[end])

    # create a matrix representation of A with the same dimensions as X*Y
    # by expanding out all data types with embedding dimension greater than 1
    if all(map(length, yidxs) .== 1)
        Areal = glrm.A # save time, but in this case we'll still have a DataFrame
    else
        Areal = zeros(m, d)
        for f=1:n
            if length(yidxs[f]) == 1
                Areal[glrm.observed_examples[f], yidxs[f]] =
                    glrm.A[glrm.observed_examples[f], f]
            else
                if isa(glrm.losses[f].domain, CategoricalDomain)
                    levels = datalevels(glrm.losses[f])
										for e in glrm.observed_examples[f]
											for ilevel in 1:length(levels)
	                        Areal[e, yidxs[f][ilevel]] =
	                            (glrm.A[e, f] == levels[ilevel] ? 1 : -1)
	                    end
									  end
                elseif isa(glrm.losses[f].domain, OrdinalDomain)
                    embed_dim = embedding_dim(glrm.losses[f])
                    mymean = mean(glrm.A[glrm.observed_examples[f], f])
                    levels = datalevels(glrm.losses[f])
										for e in glrm.observed_examples[f]
											for ilevel in 1:(length(levels)-1)
	                        Areal[e, yidxs[f][ilevel]] =
	                            (glrm.A[e, f] > levels[ilevel] ? 1 : -1)
	                    end
									  end
                else
                    error("No default mapping to real valued matrix for domains of type $typeof(glrm.losses[f].domain)")
                end
            end
        end
    end

    # standardize A, respecting missing values
    means = zeros(d)
    stds  = zeros(d)
    Astd = zeros(m, d)
    for f in 1:n
        for j in yidxs[f]
            nomissing = Areal[glrm.observed_examples[f],j]
            means[j] = mean(nomissing)
            if isnan(means[j])
                means[j] = 1
            end
            stds[j] = std(nomissing)
            if stds[j] < TOL || isnan(stds[j])
                stds[j] = 1
            end
            Astd[glrm.observed_examples[f],j] = Areal[glrm.observed_examples[f],j] .- means[j]
        end
    end
    if offset
        k -= 1
        glrm.X[end,:] = 1
        glrm.Y[end,:] = means
        if scale
            Astd = Astd ./ stds
        end
        if k <= 0
            @warn("Using an offset on a rank 1 model fits *only* the offset. To fit an offset + 1 low rank component, use k=2.")
            return glrm
        end
    end
    # options for rescaling:
    # 1) scale Astd so its mean is the same as the mean of the observations
    Astd *= m*n/sum(map(length, glrm.observed_features))
    # 2) scale columns inversely proportional to number of entries in them & so that column mean is same as mean of observations in it
    # intuition: noise in a dense column is low rank, so downweight dense columns
    # Astd *= diagm(m./map(length, glrm.observed_examples))
    # 3) scale columns proportional to scale of regularizer & so that column mean is same as mean of observations in it
    # Astd *= diagm(m./map(scale, glrm.ry))
		# ASVD = rsvd(Astd, k) - slower than built-in svds, and fails for sparse matrices
		ASVD = svds(Astd, nsv = k)[1]
    # initialize with the top k components of the SVD,
    # rescaling by the variances
    @assert(size(glrm.X, 1) >= k)
    @assert(size(glrm.X, 2) >= m)
    @assert(size(glrm.Y, 1) >= k)
    @assert(size(glrm.Y, 2) >= d)
    glrm.X[1:k,1:m] = Diagonal(sqrt.(ASVD.S))*ASVD.U' # recall X is transposed as per column major order.
    glrm.Y[1:k,1:d] = Diagonal(sqrt.(ASVD.S))*ASVD.Vt*Diagonal(stds)
    return glrm
end

include("initialize_nmf.jl")
