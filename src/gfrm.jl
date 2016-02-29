export GFRM

# todo
# * estimate lipshitz constant more reasonably
# * map GFRM to GLRM and back
# * check that PRISMA code calculates the right thing via SDP

type GFRM{L<:Loss, R<:ProductRegularizer}<:AbstractGLRM
    A                            # The data table
    losses::Array{L,1}           # Array of loss functions
    r::R                         # The regularization to be applied to U
    k::Int                       # Estimated rank of solution U (used to improve performance of PRISMA algorithm)
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
    U::AbstractArray{Float64,2}  # Representation of data in numerical space. A ≈ U = X'Y
    W::AbstractArray{Float64,2}  # Representation of data in symmetric space. W = [? U; U' ?]
end

### From GFRMs to GLRMs and back

function GFRM(glrm::GLRM; force=false, use_reg_scale=true)
    if !force # error check unless we force a conversion
    	if !isa(glrm.rx, QuadReg)
    		error("I don't know how to convert the regularization on X, $(typeof(glrm.rx)), into a regularizer for a GFRM")
    	end
    	if !(isa(glrm.ry, QuadReg) || all(map(r->isa(r, QuadReg), glrm.ry)))
    		error("I don't know how to convert the regularization on Y, $(glrm.ry), into a regularizer for a GFRM")
    	end
    end

    if use_reg_scale
    	minscale = minimum(map(scale,glrm.ry))
        maxscale = maximum(map(scale,glrm.ry))
        @assert(minscale == maxscale && minscale == scale(glrm.rx),
        	"we only know how to convert glrms with similarly scaled quadratic regularization")
        r = TraceNormReg(scale(glrm.rx)*2)
    else
    	r = TraceNormReg(1)
    end

    # translate the low rank variables X and Y into full rank variables U and W
    U = glrm.X'*glrm.Y
    W = [glrm.X glrm.Y]'*[glrm.X glrm.Y]
    return GFRM(glrm.A, glrm.losses, r, glrm.k,
		    	glrm.observed_features, glrm.observed_examples,
		    	U, W)
end

# if k=0 we use k = numerical_rank(gfrm.U)
function GLRM(gfrm::GFRM, k::Int=0; tol=1e-5)
    # we only know how to convert glrms with similarly scaled quadratic regularization
    if isa(gfrm.r, TraceNormReg)
        r = QuadReg(scale(gfrm.r)/2)
    else
    	error("I don't know how to convert $(typeof(gfrm.r)) into a regularizer for a GLRM")
    end
    u,s,v = svd(gfrm.U)
    if k == 0
    	k = sum(s.>=tol)
    else
    	k = min(k, min(size(gfrm.U)...))
    end
    X = diagm(sqrt(s[1:k]))*u[:,1:k]'
    Y = diagm(sqrt(s[1:k]))*v[:,1:k]'
    return GLRM(gfrm.A, gfrm.losses, r, r, k,
		    	observed_features = gfrm.observed_features, 
		    	observed_examples = gfrm.observed_examples,
		    	X = X, Y = Y)
end

parameter_estimate(glrm::GLRM) = (glrm.X, glrm.Y)
parameter_estimate(gfrm::GFRM) = gfrm.W
