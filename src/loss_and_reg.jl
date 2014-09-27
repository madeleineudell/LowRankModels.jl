# Loss functions and regularizers
abstract Loss

# loss functions
type quadratic<:Loss
    scale
end
quadratic() = quadratic(1)
type hinge<:Loss
    scale
end
hinge() = hinge(1)
type ordinal_hinge<:Loss
    min::Integer
    max::Integer
    scale
end
ordinal_hinge(m1,m2) = ordinal_hinge(m1,m2,1)

function grad(l::quadratic)
    return (u,a) -> (u-a)*l.scale
end

function grad(l::hinge)
    return (u,a) -> hinge_grad(u,a)
end
hinge_grad(u,a::Number) = a*u>=1 ? 0 : -a*l.scale
hinge_grad(u,a::Bool) = (2*a-1)*u>=1 ? 0 : -(2*a-1)*l.scale

function grad(l::ordinal_hinge)
    function(u,a)
        if a == l.min 
            if u>a
                return l.scale
            else 
                return 0
            end
        elseif a == l.max
            if u<a
                return -l.scale
            else
                return 0
            end
        else
            return sign(u-a) * l.scale
        end
    end
end

function evaluate(l::quadratic,u::Number,a::Number)
    l.scale*(u-a)^2
end
function evaluate(l::hinge,u::Number,a::Number)
    l.scale*max(-a*u,0)
end
function evaluate(l::ordinal_hinge,u::Number,a::Number)
    if a == l.min 
        return l.scale*max(u-a,0)
    elseif a == l.max
        return l.scale*max(a-u,0)
    else
        return l.scale*abs(u-a)
    end    
end

function variance(a::AbstractArray, l::quadratic)
    m = mean(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function variance(a::AbstractArray, l::ordinal_hinge)
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

function variance(a::AbstractArray, l::hinge)
    m = median(a)
    sum(map(ai->evaluate(l,m,ai),a))/length(a)
end

# regularizers
# regularizers r should have the method `prox` defined so that 
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)

abstract Regularizer

type quadreg<:Regularizer
    scale
end
quadreg() = quadreg(1)
function prox(r::quadreg)
    return (u,alpha) -> 1/(1+alpha*r.scale/2)*u
end
type identityreg<:Regularizer
end
function prox(r::identityreg)
    return (u,alpha) -> u
end
type nonnegative<:Regularizer
end
function prox(r::nonnegative)
    return (u,alpha) -> broadcast(max,u,0)
end
type lastcol1<:Regularizer
    r::Regularizer
end
function prox(r::lastcol1)
    return (u,alpha) -> [prox(r.r)(u[1:end-1],alpha), 1]
end
type lastcol_unpenalized<:Regularizer
    r::Regularizer
end
function prox(r::lastcol_unpenalized)
    return (u,alpha) -> [prox(r.r)(u[1:end-1],alpha), u[end]]
end
function add_offset(r::Regularizer,rt::Regularizer)
    return lastcol1(r), lastcol_unpenalized(rt)
end
type onesparse<:Regularizer
end
function prox(r::onesparse)
    return (u,alpha) -> (maxu = maximum(u); [int(ui==maxu) for ui in u])
end

# scalings
function equilibrate_variance!(losses::Array{Loss}, A)
    for i=1:size(A,2)
        vari = variance(dropna(A[:,i]), losses[i])
        if vari > 0
            losses[i].scale = 1/vari
        else
            losses[i].scale = 0
        end
    end
    return losses
end

if false
    import CVX
    function cvxjl(l::quadratic)
        return ((u,a) -> l.scale*CVX.square(u-a))
    end

    function cvxjl(l::hinge)
        return (u,a -> l.scale*CVX.pos(-a*u))
    end

    function cvxjl(l::ordinal_hinge)
        function(u,a)
            if a == l.min 
                return l.scale*CVX.pos(u-a)
            elseif a == l.max
                return l.scale*CVX.pos(a-u)
            else
                return l.scale*CVX.abs(u-a)
            end
        end
    end    

    function variance_cvx(a::AbstractArray, l::loss)
        cvxjlloss = cvxjl(l)
        u = CVX.Variable()
        p = CVX.minimize(sum([cvxjlloss(u, ai) for ai in a]))
        CVX.solve!(p)
        return p.optval/length(a)
    end

    loss2char = {identityreg=>'i', quadratic=>'q', hinge=>'b', ordinal_hinge=>'o'}
    char2loss = {'i'=>identityreg, 'q'=>quadratic, 'b'=>hinge, 'o'=>ordinal_hinge}
end