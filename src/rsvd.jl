#### randomized SVD (from Jiahao Chen, based on http://arxiv.org/pdf/0909.4061.pdf)

import Base.LinAlg.SVD

#The simplest possible randomized svd
#Inputs
#   A: input matrix
#   n: Number of singular value/vector pairs to find
#   p: Number of extra vectors to include in computation
function rsvd(A, n, p=0)
    m1, m2 = size(A)
    Q = rrange(A, n, p=p)
    rsvd_direct(A, Q)
end

#Algorithm 4.1: randomized range finder (not recommended)
#A must support size(A) and premultiply
#basis is the algorithm to compute the orthogonal basis Q
#They recommend Gram-Schmidt with double orthogonalization
#Here we use dense QR factorization using Householder reflectors
#Ω may have 'cheaper' options
#p is the oversampling parameter
function rrange(A, l::Integer; p::Integer=0, basis=_->full(qrfact!(_)[:Q]))
    p≥0 || error()
    m, n = size(A)
    l <= m || error()
    Ω = randn(n, min(l+p, m))
    Y = A*Ω
    Q = basis(Y)
    p==0 ? Q : Q[:,l]
end

#Algorithm 5.1: direct SVD
#More accurate
function rsvd_direct(A, Q)
    B=A'Q
    S=svdfact!(B)
    SVD(Q'S[:U], S[:S], S[:Vt])
end