import Base: isnan
import DataFrames: DataFrame, ncol, convert
export GLRM

# TODO: identify categoricals automatically from PooledDataArray columns

default_real_loss = HuberLoss
default_bool_loss = LogisticLoss
default_ord_loss = MultinomialOrdinalLoss

function GLRM(df::DataFrame, k::Int;
              losses = Loss[], rx = QuadReg(.01), ry = QuadReg(.01),
              offset = true, scale = false,
              prob_scale = true, NaNs_to_Missing = true)
    if NaNs_to_Missing
        df = copy(df)
        NaNs_to_Missing!(df)
    end
    if losses == Loss[] # if losses not specified, identify ordinal, boolean and real columns
        # change the wal get_reals, etc work.

        #reals, real_losses = get_reals(df)
        #bools, bool_losses = get_bools(df)
        #ordinals, ordinal_losses = get_ordinals(df)
        #easier to use just one function for this usecase.
        reals, real_losses, bools, bool_losses, ordinals, ordinal_losses = get_loss_types(df)
        A = [df[:,reals] df[:,bools] df[:,ordinals]]
        labels = [names(df)[reals]; names(df)[bools]; names(df)[ordinals]]
        losses = [real_losses; bool_losses; ordinal_losses]
    else # otherwise require one loss function per column
        A = df
        ncol(df)==length(losses) ? labels = names(df) : error("please input one loss per column of dataframe")
    end
    # identify which entries in data frame have been observed (ie are not N/A)
    obs = observations(A)
    # initialize X and Y
    X = randn(k,size(A,1))
    Y = randn(k,embedding_dim(losses))

    # form model
    rys = Array{Regularizer}(undef, length(losses))
    for i=1:length(losses)
        if isa(losses[i].domain, OrdinalDomain) && embedding_dim(losses[i])>1 #losses[i], MultinomialOrdinalLoss) || isa(losses[i], OrdisticLoss)
            rys[i] = OrdinalReg(copy(ry))
        else
            rys[i] = copy(ry)
        end
    end
    glrm = GLRM(A, losses, rx, rys, k, obs=obs, X=X, Y=Y, offset=offset, scale=scale)

    # scale model so it really computes the MAP estimator of the parameters
    if prob_scale
        prob_scale!(glrm)
    end
    return glrm, labels
end

function get_loss_types(df::DataFrame)
    m,n = size(df)
    reals = fill(false,n)
    bools = fill(false,n)
    ordinals = fill(false,n)

    for j in 1:n
        # assuming there are no columns with *all* values missing. (which would make it a non-informative column)
        t = eltype(collect(skipmissing(df[:,j]))[1])
        if(t == Float64)
            reals[j] = true
        elseif (t == Bool)
            bools[j] = true
        elseif (t == Int) || (t == Int32) || (t == Int64)
            ordinals[j] = true
        end
    end

    n1 = sum(reals)
    real_losses = Array{Loss}(undef, n1)
    for i=1:n1
        real_losses[i] = default_real_loss()
    end

    n2 = sum(bools)
    bool_losses = Array{Loss}(undef, n2)
    for i in 1:n2
        bool_losses[i] = default_bool_loss()
    end

    n3 = sum(ordinals)
    ord_idx = (1:size(df,2))[ordinals]
    maxs = zeros(n3,1)
    mins = zeros(n3,1)
    for j in 1:n3
        col = df[:,ord_idx[j]]
        try
            maxs[j] = maximum(skipmissing(col))
            mins[j] = minimum(skipmissing(col))
        catch
            nothing
        end
    end

    # set losses and regularizers
    ord_losses = Array{Loss}(undef, n3)
    for i=1:n3
        ord_losses[i] = default_ord_loss(Int(maxs[i]))
    end
    return reals,real_losses,bools,bool_losses,ordinals,ord_losses
end

function get_reals(df::DataFrame)
    m,n = size(df)
    reals = [typeof(df[:,i])<:AbstractArray{Float64,1} for i in 1:n]
    n1 = sum(reals)
    losses = Array{Loss}(undef, n1)
    for i=1:n1
        losses[i] = default_real_loss()
    end
    return reals, losses
end

function get_bools(df::DataFrame)
    m,n = size(df)
    bools = [isa(df[:,i], AbstractArray{Bool,1}) for i in 1:n]
    n1 = sum(bools)
    losses = Array{Loss}(undef, n1)
    for i=1:n1
        losses[i] = default_bool_loss()
    end
    return bools, losses
end

function get_ordinals(df::DataFrame)
    m,n = size(df)
    # there must be a better way to check types...
    ordinals = [(isa(df[:,i], AbstractArray{Int,1}) ||
                 isa(df[:,i], AbstractArray{Int32,1}) ||
                 isa(df[:,i], AbstractArray{Int64,1})) for i in 1:n]
    nord = sum(ordinals)
    ord_idx = (1:size(df,2))[ordinals]
    maxs = zeros(nord,1)
    mins = zeros(nord,1)
    for i in 1:nord
        col = df[:,ord_idx[i]]
        try
            maxs[i] = maximum(dropmissing(col))
            mins[i] = minimum(dropmissing(col))
        catch
            nothing
        end
    end

    # set losses and regularizers
    losses = Array{Loss}(undef, nord)
    for i=1:nord
        losses[i] = default_ord_loss(Int(maxs[i]))
    end
    return ordinals, losses
end
