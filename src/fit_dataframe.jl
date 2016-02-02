import Base: isnan
import DataFrames: DataFrame, DataArray, isna, dropna, array, ncol, convert, NA, NAtype

export GLRM, observations, expand_categoricals!, NaNs_to_NAs!, NAs_to_0s!

probabilistic_losses = Dict{Symbol, Any}(
    :real        => QuadLoss,
    :bool        => LogisticLoss,
    :ord         => MultinomialOrdinalLoss,
    :cat         => MultinomialLoss
)

function GLRM(df::DataFrame, k::Int, datatypes::Array{Symbol,1};
              loss_map = probabilistic_losses, 
              rx = QuadReg(.01), ry = QuadReg(.01),
              offset = true, scale = false, prob_scale = true,
              transform_data_to_numbers = true, NaNs_to_NAs = true)

    # check input
    if ncol(df)!=length(datatypes)
        error("third argument (datatypes) must have one entry for each column of data frame.")
    end
    if !all(map(dt -> dt in keys(loss_map), datatypes))
        error("data types must be either :real, :bool, :ord, or :cat")
    end

    # clean up dataframe if needed
    A = copy(df)
    if NaNs_to_NAs    
        NaNs_to_NAs!(A)
    end

    # define loss functions for each column
    losses = Array(Loss, ncol(A))
    for j=1:ncol(df)
        loss = loss_map[datatypes[j]]
        if transform_data_to_numbers
            map_to_numbers!(A, j, datatypes[j])
        end
        losses[j] = pick_loss(loss, A[j])
    end

    # identify which entries in data frame have been observed (ie are not N/A)
    obs = observations(df)

    # form model
    rys = Array(Regularizer, length(losses))
    for i=1:length(losses)
        if isa(losses[i], MultinomialOrdinalLoss) || isa(losses[i], OrdisticLoss)
            rys[i] = OrdinalReg(copy(ry))
        else
            rys[i] = copy(ry)
        end
    end
    glrm = GLRM(A, losses, rx, rys, k, obs=obs, offset=offset, scale=scale)
    
    # scale model so it really computes the MAP estimator of the parameters
    if prob_scale
        prob_scale!(glrm)
    end

    return glrm
end

## transform data to numbers

function map_to_numbers!(df, j::Int, datatype::Symbol)
    # easy case
    if datatype == :real
        if all(xi -> isa(xi, Number), df[j][!isna(df[j])])
            return df[j]
        else
            error("column contains non-numerical values")
        end
    end
    
    # harder cases
    col = copy(df[j])
    levels = Set(col[!isna(col)])
    if datatype == :bool
        if length(levels)>2
            error("Boolean variable should have at most two levels")
        end
        colmap = Dict{Any,Int}(zip(sort(collect(levels)), [-1,1][1:length(levels)]))
    elseif datatype == :cat || datatype == :ord
        colmap = Dict{Any,Int}(zip(sort(collect(levels)), 1:length(levels)))
    else
        error("datatype $datatype not recognized")
    end
    df[j] = DataArray(Int, length(df[j]))
    for i in 1:length(col)
        if !isna(col[i])
            df[j][i] = colmap[col[i]]
        end
    end
    return df[j]
end

function map_to_numbers!(df, j::Int, loss::Type{QuadLoss})
    if all(xi -> isa(xi, Number), df[j][!isna(df[j])])
        return df[j]
    else
        error("column contains non-numerical values")
    end
end

function map_to_numbers!(df, j::Int, loss::Type{LogisticLoss})
    col = copy(df[j])
    levels = Set(col[!isna(col)])
    if length(levels)>2
        error("Boolean variable should have at most two levels")
    end
    colmap = Dict{Any,Int}(zip(sort(collect(levels)), [-1,1][1:length(levels)]))
    df[j] = DataArray(Int, length(df[j]))
    for i in 1:length(col)
        if !isna(col[i])
            df[j][i] = colmap[col[i]]
        end
    end
    return df[j]
end

function map_to_numbers!(df, j::Int, loss::Type{MultinomialLoss})
    col = copy(df[j])
    levels = Set(col[!isna(col)])
    colmap = Dict{Any,Int}(zip(sort(collect(levels)), 1:length(levels)))
    df[j] = DataArray(Int, length(df[j]))
    for i in 1:length(col)
        if !isna(col[i])
            df[j][i] = colmap[col[i]]
        end
    end
    return df[j]
end

function map_to_numbers!(df, j::Int, loss::Type{MultinomialOrdinalLoss})
    col = copy(df[j])
    levels = Set(col[!isna(col)])
    colmap = Dict{Any,Int}(zip(sort(collect(levels)), 1:length(levels)))
    df[j] = DataArray(Int, length(df[j]))
    for i in 1:length(col)
        if !isna(col[i])
            df[j][i] = colmap[col[i]]
        end
    end
    return df[j]
end

## sanity check the choice of loss

function pick_loss(l::Type{QuadLoss}, col)
    return l()
end

function pick_loss(l::Type{LogisticLoss}, col)
    if all(xi -> isna(xi) || xi in [-1,1], col)
        return l()
    else
        error("LogisticLoss can only be used on data taking values in {-1, 1}")
    end
end

function pick_loss(l::Type{MultinomialLoss}, col)
    if all(xi -> isna(xi) || (isa(xi, Int) && xi >= 1), col)
        return l(maximum(col[!isna(col)]))
    else
        error("MultinomialLoss can only be used on data taking positive integer values")
    end
end

function pick_loss(l::Type{MultinomialOrdinalLoss}, col)
    if all(xi -> isna(xi) || (isa(xi, Int) && xi >= 1), col)
        return l(maximum(col[!isna(col)]))
    else
        error("MultinomialOrdinalLoss can only be used on data taking positive integer values")
    end
end

observations(da::DataArray) = df_observations(da)
observations(df::DataFrame) = df_observations(df)
function df_observations(da)
    obs = @compat Tuple{Int, Int}[]
    m,n = size(da)
    for j=1:n # follow column-major order. First element of index in innermost loop
        for i=1:m
            if !isna(da[i,j])
                push!(obs,(i,j))
            end
        end
    end
    return obs
end

# NAs in the data frame will be replaced by the number `z`
function df2array(df::DataFrame, z::Number)
    A = zeros(size(df))
    for i=1:size(A,2)
        if issubtype(typeof(df[:,i]), Array)
            A[:,i] = df[:,i]
        elseif typeof(df[i]) == Bool
            A[:,i] = convert(Array, (2*df[i]-1), z)
        else
            A[:,i] = convert(Array, df[i], z)
        end            
    end
    return A
end
df2array(df::DataFrame) = df2array(df, 0)

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

function expand_categoricals!(df::DataFrame,categoricals::Array)
    categoricalidxs = map(y->df.colindex[y], categoricals)
    # create one boolean column for each level of categorical column
    for col in categoricals
        levels = sort(unique(df[:,col]))
        for level in levels
            if !isna(level)
                colname = symbol(string(col)*"="*string(level))
                df[colname] = (df[:,col] .== level)
            end
        end
    end
    # remove the original categorical columns
    return df[:, filter(x->(!(x in categoricals)), names(df))]
end

# convert NaNs to NAs
isnan(x::NAtype) = false
isnan(x::AbstractString) = false
function NaNs_to_NAs!(df::DataFrame)
    m,n = size(df)
    for j=1:n # follow column-major order. First element of index in innermost loop
        for i=1:m
            if isnan(df[i,j])
                df[i,j] = NA
            end
        end
    end
    return df
end

function NAs_to_0s!(df::DataFrame)
    m,n = size(df)
    for j=1:n # follow column-major order. First element of index in innermost loop
        for i=1:m
            if isna(df[i,j])
                df[i,j] = 0
            end
        end
    end
    return df
end