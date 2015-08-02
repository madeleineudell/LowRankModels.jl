import StatsBase.sample, StatsBase.wsample
export init_kmeanspp!, init_svd!, init_nnmf!

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
        glrm.X[end,:] = 1
        glrm.Y[end,:] = means
        Astd = Ademeaned*diagm(1./stds)
    else
        # i'm not sure why you'd want to do this, unless you're sure the data was already demeaned,
        # or possibly to cope with regularization
        k = glrm.k
        Astd = A*diagm(1./stds)
    end
    # options for rescaling:
    # 1) scale Astd so its mean is the same as the mean of the observations
    Astd *= m*n/sum(map(length, glrm.observed_features))
    # 2) scale columns inversely proportional to number of entries in them & so that column mean is same as mean of observations in it
    # intuition: noise in a dense column is low rank, so downweight dense columns
    # Astd *= diagm(m./map(length, glrm.observed_examples))
    # 3) scale columns proportional to scale of regularizer & so that column mean is same as mean of observations in it
    # Astd *= diagm(m./map(scale, glrm.ry))
    ASVD = rsvd(Astd, k)
    # initialize with the top k components of the SVD,
    # rescaling by the variances
    glrm.X[1:k,1:m] = diagm(sqrt(ASVD[:S]))*ASVD[:U]' # recall X is transposed as per column major order.
    glrm.Y[1:k,1:n] = diagm(sqrt(ASVD[:S]))*ASVD[:Vt]*diagm(stds)
    return glrm
end

function init_nnmf!(glrm::GLRM; scaling=true, variant=:nndsvd)
    # NNDSVD initialization:
    #    Boutsidis C, Gallopoulos E (2007). SVD based initialization: A head
    #    start for nonnegative matrix factorization. Pattern Recognition 
    m,n = size(glrm.A)

    # only initialize based on observed entries
    A_sparse = zeros(size(glrm.A))
    for i = 1:n
        A_sparse[glrm.observed_examples[i],i] = glrm.A[glrm.observed_examples[i],i]
    end

    # scale all columns by the Loss.scale parameter
    if scaling
        for i = 1:n
            A_sparse[:,i] ./= glrm.losses[i].scale
        end
    end

    # compute svd
    U,s,V = svd(A_sparse, thin=true)

    # determine how to initialize negative values
    z0 = variant == :nndsvd ? 0.0 :
         variant == :nndsvd_a ? mean(A_sparse) : 
         variant == :nndsvd_ar ? (mean(A_sparse)*0.01) :
         error("NNDSVD variant not recognized")

    # main loop
    for j = 1:glrm.k
        uj = view(U,:,j)
        vj = view(V,:,j)
        u_pnrm, u_nnrm = posnegnorm(uj)
        v_pnrm, v_nnrm = posnegnorm(vj)
        mp = v_pnrm * u_pnrm
        mn = v_nnrm * u_nnrm

        # randomization for nndsvd_ar variant
        zj = z0
        if variant == :nndsvd_ar
            zj *= rand()
        end

        # scale X and Y
        if mp >= mn
            scalepos!(view(glrm.X,j,:), uj, 1 / u_pnrm, zj) # Remember X is transposed (in column-major order)
            scalepos!(view(glrm.Y,j,:), vj, s[j] * mp / v_pnrm, zj)
        else
            scaleneg!(view(glrm.X,j,:), uj, 1 / u_nnrm, zj)  # Remember X is transposed (in column-major order)
            scaleneg!(view(glrm.Y,j,:), vj, s[j] * mn / v_nnrm, zj)
        end            
    end

	return glrm
end

## The following functions are reproduced from the
## NMF.jl package (https://github.com/JuliaStats/NMF.jl)

# compute separate norms for pos and neg elements of a vector
function posnegnorm{T}(x::AbstractArray{T})
    pn = zero(T)
    nn = zero(T)
    for i = 1:length(x)
        @inbounds xi = x[i]
        if xi > zero(T)
            pn += abs2(xi)
        else
            nn += abs2(xi)
        end
    end
    return (sqrt(pn), sqrt(nn))
end

# y = x * c; setting negative elements to v0
function scalepos!{T<:Number}(y, x, c::T, v0::T)
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi > zero(T)
            y[i] = xi * c
        else
            y[i] = v0
        end
    end
end

# y = -x * c; setting negative elements to v0
function scaleneg!{T<:Number}(y, x, c::T, v0::T)
    @inbounds for i = 1:length(y)
        xi = x[i]
        if xi < zero(T)
            y[i] = - (xi * c)
        else
            y[i] = v0
        end
    end
end
