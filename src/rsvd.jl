#### randomized SVD (from Jiahao Chen, based on http://arxiv.org/pdf/0909.4061.pdf)

import Base.LinAlg.SVD

#The simplest possible randomized svd
#Inputs
#   A: input matrix
#   n: Number of singular value/vector pairs to find
#   p: Number of extra vectors to include in computation
function rsvd(A, n, p=0)
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
function rrange(A, l::Integer; p::Integer=0)
    p≥0 || error()
    m, n = size(A)
    l <= m || error("Cannot find $l linearly independent vectors of $m x $n matrix")
    Ω = randn(n, l+p)
    Y = A*Ω
    Q = full(qrfact!(Y)[:Q])
    Q = p==0 ? Q : Q[:,1:l] #TODO slicing of QRCompactWYQ NOT IMPLEMENTED
    @assert l==size(Q, 2)
    Q
end

#Algorithm 5.1: direct SVD
#More accurate
function rsvd_direct(A, Q)
    B=Q'A
    S=svdfact!(B)
    SVD(Q*S[:U], S[:S], S[:Vt])
end