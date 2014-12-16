function nearest_center(centers, point)


# kmeans++, but for missing data
# we make sure never to look at "unobserved" entries in A
# so that models can be honestly cross validated, for example
function init_kmeanspp(glrm::GLRM)
	m,n = size(glrm.A)
	k = glrm.k
	possible_centers = set(1:m)
	glrm.Y = randn(k,n)
	# assign first center randomly
	i = sample(1:m)
	setdiff!(possible_centers, i)
	glrm.Y[1,glrm.observed_features[i]] = glrm.A[i,glrm.observed_features[i]]
	# assign next centers one by one
	for l=1:k-1
		min_dists = zeros(m)
		for i in possible_centers
			d = zeros(l)
			for j in glrm.observed_features[i]
				for ll=1:l
					d[ll] += evaluate(glrm.losses[j], glrm.Y[ll,j], glrm.A[i,j])
				end
			end
			min_dists[i] = minimum(d)
		end
        furthest_index = wsample(1:m,min_dists)
		glrm.Y[l+1,glrm.observed_features[furthest_index]] = glrm.A[furthest_index,glrm.observed_features[furthest_index]]
	return glrm
end

function intialize_centers(points,centers)
    # Iteratively pick points to be clusters based on how far a given point is away
    # already selected cluster centers.
    # @points NxP Array of Float of points to be clustered
    # @clust MxP Array: centers where M is the number of clusters
    # output: MxP Array of new cluster centers
    clust_centers = copy(centers)
    numdims = size(points)
    clust_centers[1,:] = points[rand(1:numdims[1]),:]
    dists = zeros(numdims[1])
    for center = 1:(size(clust_centers,1)-1)
        point_dist = 0
        for dist_entry in 1:size(dists,1)
            dists[dist_entry] = nearest_center(points[dist_entry,:],clust_centers[center,:,:][1,:])[2]
            point_dist += dists[dist_entry]
        end
        furthest_index = wsample(1:numdims[1],dists./point_dist)
        clust_centers[center+1,:] = points[furthest_index,:]
    end
    return clust_centers;
end

function init_svd(glrm::GLRM)
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

function init_nnmf(glrm::GLRM)
	return glrm
end