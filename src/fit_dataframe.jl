#module FitDataFrame

import DataFrames: DataFrame, DataArray, isna, dropna, array

export GLRM, observations, expand_categoricals, add_offset!, equilibrate_variance!

max_ordinal_levels = 9

function df2array(df::DataFrame, z::Number)
    A = zeros(size(df))
    for i=1:size(A,2)
        if typeof(df[i]) == Bool
            A[:,i] = array((2*df[i]-1),z)
        else
            A[:,i] = array(df[i],z)
        end            
    end
    return A
end
df2array(df::DataFrame) = df2array(df, 0)

function observations(df::DataFrame)
    obs = (Int32, Int32)[]
    m,n = size(df)
    for i=1:m
        for j=1:n
            if !isna(df[i,j])
                push!(obs,(i,j))
            end
        end
    end
    return obs
end

function get_reals(df::DataFrame, loss=huber)
    m,n = size(df)
    reals = [typeof(df[i])<:DataArray{Float64,1} for i in 1:n]
    n1 = sum(reals)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = loss()
    end
    return reals, losses
end

function get_bools(df::DataFrame, loss=hinge)
    m,n = size(df)
    bools = [(typeof(df[i])<:DataArray{Bool,1} || all([x in [-1,1] for x in unique(df[i][!isna(df[i])])])) for i in 1:n]
    n1 = sum(bools)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = loss()
    end
    return bools, losses
end

function get_ordinals(df::DataFrame)
    m,n = size(df)
    ordinals = [typeof(df[i])<:DataArray{Int,1} for i in 1:n]
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
    if loss==ordinal_hinge
        for i=1:nord
            losses[i] = ordinal_hinge(mins[i],maxs[i])
        end
    else
        error("get_ordinals not implemented for losses of type $loss")
    end
    return ordinals, losses
end

function expand_categoricals(df::DataFrame,categoricals::Array)
    categoricalidxs = map(y->df.colindex[y], categoricals)
    # create one boolean column for each level of categorical column
    for col in categoricals
        levels = unique(df[:,col])
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

# scalings and offsets
function add_offset!(glrm::GLRM)
    glrm.rx, glrm.ry = lastentry1(glrm.rx), map(lastentry_unpenalized, glrm.ry)
    return glrm
end
function equilibrate_variance!(glrm::GLRM)
    for i=1:size(glrm.A,2)
        nomissing = glrm.A[glrm.observed_examples[i],i]
        if length(nomissing)>0
            varlossi = avgerror(nomissing, glrm.losses[i])
            varregi = var(nomissing) # TODO make this depend on the kind of regularization; this assumes quadratic
        else
            varlossi = 1
            varregi = 1
        end
        if varlossi > 0
            # rescale the losses and regularizers for each column by the inverse of the empirical variance
            scale!(glrm.losses[i], scale(glrm.losses[i])/varlossi)
        end
        if varregi > 0
            scale!(glrm.ry[i], scale(glrm.ry[i])/varregi)
        end
    end
    return glrm
end

function GLRM(df::DataFrame, k::Integer;
              losses = {:real=>huber, :bool=>hinge, :ord=>hinge}, 
              rx = quadreg(.01), ry = quadreg(.01),
              offset = true, scale = true)
    # identify ordinal, boolean and real columns
    if losses == None
        reals, real_losses = get_reals(df)
        bools, bool_losses = get_bools(df)
        ordinals, ordinal_losses = get_ordinals(df)

        A = [df[reals] df[bools] df[ordinals]]
        labels = [names(df)[reals], names(df)[bools], names(df)[ordinals]]
        losses = [real_losses, bool_losses, ordinal_losses]
    elseif isa(losses, Array)
        # otherwise one loss function per column
        ncol(df)==length(losses) ? labels = names(df) : error("please input one loss per column of dataframe")
    elseif isa(losses, Dict)
        reals, real_losses = get_reals(df, losses[:real])
        bools, bool_losses = get_bools(df, losses[:bool])
        ordinals, ordinal_losses = get_ordinals(df, losses[:ord])

        A = [df[reals] df[bools] df[ordinals]]
        labels = [names(df)[reals], names(df)[bools], names(df)[ordinals]]
        losses = [real_losses, bool_losses, ordinal_losses]
    end

    # identify which entries in data frame have been observed (ie are not N/A) and form model
    obs = observations(A)
    glrm = GLRM(df2array(A), obs, losses, rx, ry, k)
    
    # scale losses (and regularizers) so they all have equal variance
    if scale
        equilibrate_variance!(glrm)
    end
    # don't penalize the offset of the columns
    if offset
        add_offset!(glrm)
    end

    # return model
    return glrm, labels
end
# this replaces all NAs with zeros --- deprecated
# GLRM(df::DataFrame,args...) = GLRM(df2array(df,0),args...)

#end
