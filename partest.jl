function doparallelstuff(m = 10, n = 20)
    # initialize variables
    localX = Base.shmem_rand(m; pids=procs())
    localY = Base.shmem_rand(n; pids=procs())
    localf = [x->i+sum(x) for i=1:m]
    localg = [x->i+sum(x) for i=1:n]

    # broadcast variables to all worker processes
    @sync begin
        for i in procs(localX)
            remotecall(i, x->(global X; X=x; nothing), localX)
            remotecall(i, x->(global Y; Y=x; nothing), localY)
            remotecall(i, x->(global f; f=x; nothing), localf)
            remotecall(i, x->(global g; g=x; nothing), localg)
        end
    end

    # compute
    for iteration=1:1
        @everywhere for i=localindexes(X)
            X[i] = f[i](Y)
        end
        @everywhere for j=localindexes(Y)
            Y[j] = g[j](X)
        end
    end
end

doparallelstuff()