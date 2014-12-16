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

function init_svd!(glrm::GLRM)
	m,n = size(glrm.A)
	k = glrm.k
	B = zeros(m,n)
	for i=1:m
		for j in glrm.observed_features[i]
			B[i,j] = A[i,j]
		end
	end
	u,s,v = svd(B)
	glrm.X = u[:,1:k]*diagm(sqrt(s[1:k]))
	glrm.Y = diagm(sqrt(s[1:k]))*v[:,1:k]'
	return glrm
end

function init_nnmf!(glrm::GLRM)
	return glrm
end