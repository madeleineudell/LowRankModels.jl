@everywhere begin
global X
global Y
global f
global g
end

function doparallelstuff(m = 10, n = 20)
    # initialize variables
    localX = Base.shmem_rand(m)
    localY = Base.shmem_rand(n)
    localf = [x->i+sum(x) for i=1:m]
    localg = [x->i+sum(x) for i=1:n]

    # broadcast variables to all worker processes
    @parallel for i=workers()
        global X = localX
        global Y = localY
        global f = localf
        global g = localg
    end
    # give variables same name on master
    X,Y,f,g = localX,localY,localf,localg

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