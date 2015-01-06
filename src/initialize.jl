import StatsBase.sample
export init_kmeanspp!, init_svd!

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

function init_svd!(glrm::GLRM; offset=false)
	m,n = size(glrm.A)
	if offset
		k = glrm.k-1
		# medians of columns
		medians = median(glrm.A,1)
		A = glrm.A .- medians
		glrm.X[:,end] = 1
		glrm.Y[end,:] = medians
	else
		k = glrm.k
		A = glrm.A
	end
	B = zeros(m,n)
	for i=1:m
		for j in glrm.observed_features[i]
			B[i,j] = glrm.A[i,j]
		end
	end
	# scale B so its mean is the same as the mean of the observations
	B *= m*n/sum(map(length, glrm.observed_features))
	u,s,v = svd(B)
	glrm.X[1:m,1:k] = u[:,1:k]*diagm(sqrt(s[1:k]))
	glrm.Y[1:k,1:n] = diagm(sqrt(s[1:k]))*v[:,1:k]'
	return glrm
end

function init_nnmf!(glrm::GLRM)
	return glrm
end