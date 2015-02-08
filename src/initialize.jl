import StatsBase.sample
export init_kmeanspp!, init_svd!

include("rsvd.jl")

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

function init_svd!(glrm::GLRM; offset=true, TOL = 1e-10)
    m,n = size(glrm.A)
    # standardize A, respecting missing values
    means = zeros(n)
    stds  = zeros(n)
    Ademeaned = zeros(size(glrm.A))
    for i=1:n
        nomissing = glrm.A[glrm.observed_examples[i],i]
        means[i] = mean(nomissing)
        if isnan(means[i])
            means[i] = 1
        end
        stds[i] = std(nomissing)
        if stds[i] < TOL || isnan(stds[i])
            stds[i] = 1
        end
        Ademeaned[glrm.observed_examples[i],i] = glrm.A[glrm.observed_examples[i],i] - means[i]
    end
    if offset
        k = glrm.k-1
        glrm.X[:,end] = 1
        glrm.Y[end,:] = means
        Astd = Ademeaned*diagm(1./stds)
    else
        # i'm not sure why you'd want to do this, unless you're sure the data was already demeaned,
        # or possibly to cope with regularization
        k = glrm.k
        Astd = A*diagm(1./stds)
    end
    # TODO: might also want to remove entries in columns that have many fewer missing values than others
    # intuition: noise in a dense column is low rank
    # scale Astd so its mean is the same as the mean of the observations
    Astd *= m*n/sum(map(length, glrm.observed_features))
    S = rsvd(Astd, k, 3)
    u,s,v = S[:U], S[:S], S[:V]
    # initialize with the top k components of the SVD,
    # rescaling by the variances
    glrm.X[1:m,1:k] = u[:,1:k]*diagm(sqrt(s[1:k]))
    glrm.Y[1:k,1:n] = diagm(sqrt(s[1:k]))*v[:,1:k]'*diagm(stds)
    return glrm
end

function init_nnmf!(glrm::GLRM)
	return glrm
end

function init_convex(glrm::GLRM; delta = 1e-3, maxiter = 100, verbose = false)
    # extract parameters from glrm
    @assert all([glrm.rx.scale == glrm.ry[i].scale for i=1:length(glrm.ry)])
    gamma = 1/glrm.rx.scale/2
    # this should be 1/sqrt(max(m,n)) for RPCA
    k = glrm.k
    A = glrm.A
    m, n = size(A)
    
    # recommended by Candes for robust PCA
    mu = n^2/4/vecnorm(A, 1)
    
    S = A 
    L = zeros(size(A))
    Y = zeros(size(A))
    normAF = vecnorm(A)
    kmax = k*2
    # TODO: pick initial lambda that makes the solution extremely low rank
    for lambda = linspace(.1/sqrt(max(m,n)), gamma, 5)
        if verbose println("lambda = $lambda") end
        # consider whether resetting Y to 0 is mathematically necessary; it's definitely slower
        # Y = zeros(size(A))
        for i = 1:maxiter
            if verbose && i%10 == 0 println("\titeration $i") end
            # TODO: replace this with the prox on the observed entries
            L = soft_threshold_singular_values(A - S + Y ./ mu, mu; kmax = kmax)
            S = soft_threshold(A - L + Y ./ mu, lambda * mu)
            Y = Y + mu * (A - L - S)
            if vecnorm(A - L - S) <= delta*normAF
                 break
            end
        end
        if verbose println("\trank(L) is ", rank(L)) end
        if rank(L) > k
            if verbose println("lets go nonconvex!") end
            break
        end
    end
    u,s,v = svd(L)
    glrm.X = u[:,1:glrm.k]*diagm(sqrt(s[1:glrm.k])); glrm.Y = diagm(sqrt(s[1:glrm.k]))*v[:,1:glrm.k]'
    return glrm.X, glrm.Y
end