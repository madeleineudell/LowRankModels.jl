import NMF.nndsvd

function init_nndsvd!(glrm::GLRM; scale::Bool=true, zeroh::Bool=false,
                      variant::Symbol=:std, max_iters::Int=0)
    # NNDSVD initialization:
    #    Boutsidis C, Gallopoulos E (2007). SVD based initialization: A head
    #    start for nonnegative matrix factorization. Pattern Recognition
    m,n = size(glrm.A)

    # only initialize based on observed entries
    A_init = zeros(m,n)
    for i = 1:n
        A_init[glrm.observed_examples[i],i] = glrm.A[glrm.observed_examples[i],i]
    end

    # scale all columns by the Loss.scale parameter
    if scale
        for i = 1:n
            A_init[:,i] .*= glrm.losses[i].scale
        end
    end

    # run the first nndsvd initialization
    W,H = nndsvd(A_init, glrm.k, zeroh=zeroh, variant=variant)
    glrm.X = W'
    glrm.Y = H

    # If max_iters>0 do a soft impute for the missing entries of A.
    #   Iterate: Estimate missing entries of A with W*H
    #            Update (W,H) nndsvd estimate based on new A
    for iter = 1:max_iters
        # Update missing entries of A_init
        for j = 1:n
            for i = setdiff(1:m,glrm.observed_examples[j])
                A_init[i,j] = dot(glrm.X[:,i],glrm.Y[:,j])
            end
        end
        # Re-estimate W and H
        W,H = nndsvd(A_init, glrm.k, zeroh=zeroh, variant=variant)
        glrm.X = W'
        glrm.Y = H
    end
end
