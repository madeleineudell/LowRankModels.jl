export GFRM

# todo
# * estimate lipshitz constant more reasonably
# * map GFRM to GLRM and back
# * check that PRISMA code calculates the right thing via SDP

type GFRM{L<:Loss, R<:ProductRegularizer}<:AbstractGLRM
    A                            # The data table
    losses::Array{L,1}           # Array of loss functions
    r::R                         # The regularization to be applied to U
    k::Int                       # Estimated rank of solution U
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
    U::AbstractArray{Float64,2}  # Representation of data in numerical space. A ≈ U = X'Y
    W::AbstractArray{Float64,2}  # Representation of data in symmetric space. W = [? U; U' ?]
end

### From GFRMs to GLRMs and back

function GFRM(glrm::GLRM)
    # we only know how to convert glrms with similarly scaled quadratic regularization
    # @show isa(glrm.rx, QuadReg)
    # @show isa(glrm.ry, QuadReg)
    # @show all(map(r->isa(r, QuadReg), glrm.ry))
    if isa(glrm.rx, QuadReg) && 
    	(isa(glrm.ry, QuadReg) || all(map(r->isa(r, QuadReg), glrm.ry)))
        minscale = minimum(map(scale,glrm.ry))
        maxscale = maximum(map(scale,glrm.ry))
        @assert(minscale == maxscale && minscale == scale(glrm.rx),
        	"we only know how to convert glrms with similarly scaled quadratic regularization")
        r = TraceNormReg(glrm.rx.scale*2)
    else 
    	error("we don't know how to convert non-quadratic regularization on the factors X and Y into a convex regularizer on X'*Y")
    end
    U = glrm.X'*glrm.Y
    W = [glrm.X glrm.Y]'*[glrm.X glrm.Y]
    return GFRM(glrm.A, glrm.losses, r, glrm.k,
		    	glrm.observed_features, glrm.observed_examples,
		    	U, W)
end

function GLRM(gfrm::GFRM)
    # we only know how to convert glrms with similarly scaled quadratic regularization
    if isa(gfrm.r, TraceNormReg)
        r = TraceNormReg(glrm.rx.scale*2)
    end
    U = X'*Y
    W = [X Y]'*[X Y]
    return GFRM(glrm.A, glrm.losses, r, glrm.k,
		    	glrm.observed_features, glrm.observed_examples,
		    	U, W)
end
