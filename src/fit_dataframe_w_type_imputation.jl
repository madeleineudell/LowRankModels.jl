import Base: isnan
import DataFrames: DataFrame, DataArray, isna, dropna, array, ncol, convert, NA, NAtype

export GLRM

# TODO: identify categoricals automatically from PooledDataArray columns

function GLRM(df::DataFrame, k::Int;
              losses = Loss[], rx = QuadReg(.01), ry = QuadReg(.01),
              offset = true, scale = true, NaNs_to_NAs = false)
    if NaNs_to_NAs
        df = copy(df)
        NaNs_to_NAs!(df)
    end
    if losses == Loss[] # if losses not specified, identify ordinal, boolean and real columns
        reals, real_losses = get_reals(df)
        bools, bool_losses = get_bools(df)
        ordinals, ordinal_losses = get_ordinals(df)
        A = [df[reals] df[bools] df[ordinals]]
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
    Y = randn(k,size(A,2))
    # form model
    glrm = GLRM(A, losses, rx, ry, k, obs=obs, X=X, Y=Y, offset=offset, scale=scale)
    return glrm, labels
end

function get_reals(df::DataFrame)
    m,n = size(df)
    reals = [typeof(df[i])<:AbstractArray{Float64,1} for i in 1:n]
    n1 = sum(reals)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = QuadLoss()
    end
    return reals, losses
end

function get_bools(df::DataFrame)
    m,n = size(df)
    bools = [isa(df[i], AbstractArray{Bool,1}) for i in 1:n]
    n1 = sum(bools)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = HingeLoss()
    end
    return bools, losses
end

function get_ordinals(df::DataFrame)
    m,n = size(df)
    # there must be a better way to check types...
    ordinals = [(isa(df[i], AbstractArray{Int,1}) || 
                 isa(df[i], AbstractArray{Int32,1}) || 
                 isa(df[i], AbstractArray{Int64,1})) for i in 1:n]
    nord = sum(ordinals)
    ord_idx = (1:size(df,2))[ordinals]
    maxs = zeros(nord,1)
    mins = zeros(nord,1)
    for i in 1:nord
        col = df[ord_idx[i]]
        try
            maxs[i] = maximum(dropna(col))
            mins[i] = minimum(dropna(col))
        end
    end

    # set losses and regularizers
    losses = Array(Loss,nord)
    for i=1:nord
        losses[i] = OrdinalHinge(mins[i],maxs[i])
    end
    return ordinals, losses
end