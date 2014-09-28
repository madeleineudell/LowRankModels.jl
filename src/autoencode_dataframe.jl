using GLRM, DataFrames

export autoencode_dataframe, observations

max_ordinal_levels = 9

function df2array(df::DataFrame,z::Number)
    A = zeros(size(df))
    for i=1:size(A,2)
        A[:,i] = array(df[i],z)
    end
    return A
end
function observations(df::DataFrame)
    obs = {}
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
function check_observations(obs)
    # XXX make sure we have at least one observation of each feature and example
    true
end
GLRM(df::DataFrame,args...) = GLRM(df2array(df,0),args...)

function get_reals(df::DataFrame)
    reals = [typeof(df[i])<:DataArray{Float64,1} for i in 1:ncol(df)]
    n1 = sum(reals)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = quadratic()
    end
    return reals, losses
end

function get_bools(df::DataFrame)
    bools = [typeof(df[i])<:DataArray{Bool,1} for i in 1:ncol(df)]
    n1 = sum(bools)
    losses = Array(Loss,n1)
    for i=1:n1
        losses[i] = hinge()
    end
    return bools, losses
end

function get_ordinals(df::DataFrame)
    ordinals = [typeof(df[i])<:DataArray{Int32,1} for i in 1:ncol(df)]
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
        losses[i] = ordinal_hinge(mins[i],maxs[i])
    end
    return ordinals, losses
end

function autoencode_dataframe(df::DataFrame, k::Integer; 
                              losses = None, r = quadreg(.1), rt = quadreg(.1), 
                              offset = true, scale = true)
    # identify ordinal, boolean and real columns
    if losses == None
        reals, real_losses = get_reals(df)
        bools, bool_losses = get_bools(df)
        ordinals, ordinal_losses = get_ordinals(df)

        A = [df[reals] df[bools] df[ordinals]]
        labels = [names(df)[reals], names(df)[bools], names(df)[ordinals]]
        losses = [real_losses, bool_losses, ordinal_losses]
    else
        # otherwise one loss function per column
        ncol(df)==length(losses) ? labels = names(df) : error("please input one loss per column of dataframe")
    end

    # identify which entries in data frame have been observed (ie are not N/A)
    obs = observations(A)

    # scale losses so they all have equal variance
    if scale
        equilibrate_variance!(losses, A)
    end
    # don't penalize the offset of the columns
    if offset
        r, rt = add_offset(r, rt)
    end

    # go!
    glrm = GLRM(A, obs, losses, rt, r, k)
    X,Y,ch = autoencode(glrm)
    return X,Y,labels,ch
end