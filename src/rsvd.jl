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

#Algorithm 4.4: randomized subspace iteration
#A must support size(A), multiply and transpose multiply
#p is the oversampling parameter
#q controls the accuracy of the subspace found; it is the "number of power iterations"
#A good heuristic is that when the original scheme produces a basis whose
#approximation error is within a factor C of the optimum, the power scheme produces
#an approximation error within C^(1/(2q+1)) of the optimum.
function rrange(A, l::Integer; p::Integer=5, q::Integer=3)
    p≥0 || error()
    m, n = size(A)
    l <= m || error("Cannot find $l linearly independent vectors of $m x $n matrix")
    Ω = randn(n, l+p)
    Q = q_from_qr(A*Ω)
    for t=1:q
        Q = q_from_qr(A'*Q)
        Q = q_from_qr(A*Q)
    end
    Q = p==0 ? Q : Q[:,1:l]
end

function q_from_qr(Y, l::Integer=-1)
    Q = full(qrfact!(Y)[:Q])
    Q = l<0 ? Q : Q[:,1:l]
end

#Algorithm 5.1: direct SVD
#More accurate
function rsvd_direct(A, Q)
    B=Q'A
    S=svdfact!(B)
    SVD(Q*S[:U], S[:S], S[:Vt])
end