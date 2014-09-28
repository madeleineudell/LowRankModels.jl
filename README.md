# GLRM.jl

[![Build Status](https://travis-ci.org/madeleineudell/GLRM.jl.svg?branch=master)](https://travis-ci.org/madeleineudell/GLRM.jl)

GLRM.jl is a julia package for modeling and fitting generalized low rank models (GLRMs).
GLRMs model a data array by a low rank matrix, and
include many well known models in data analysis, such as 
principal components analysis (PCA), matrix completion, robust PCA,
nonnegative matrix factorization, k-means, and many more.

GLRM.jl makes it easy to mix and match loss functions and regularizers
to construct a model suitable for a particular data set.
In particular, it supports 

* using different loss functions for different columns of the data array, 
  which can be very useful when data types are heterogeneous 
  (eg, real, boolean, and ordinal columns);
* data tables with many missing (unobserved) entries
* adding offsets and scalings to the model without destroying sparsity,
  which can be very useful when the data is poorly scaled.

# Generalized Low Rank Models

GLRMs form a low rank model for tabular data $A$ with $m$ rows and $n$ columns, 
which can be input as an array or any array-like object (for example, a data frame).
It is fine if only some of the entries $(i,j) \in \Omega$ have been observed 
(i.e., the others are missing or `NA`); the GLRM will only be fit on the observed entries.
The desired model is specified by choosing a rank $k$ for the model,
an array of loss functions $L_j$, and two regularizers, $r$ and $rt$.
The data is modeled as $XY$, where $X \in R^{m \times k}$ and $Y \in R^{k \times n}$.
$X$ and $Y$ are found by solving the optimization problem
$$
\begin{array}{ll}
\mbox{minimize} & \sum_{(i,j) \in \Omega} L_{ij}(x_i y_j, A_{ij}) 
+ \sum_{i=1}^m r_i(x_i) + \sum_{j=1}^n \tilde r_j(y_j),
\end{array}
$$

The basic type used by GLRM.jl is (unsurprisingly), the GLRM. To form a GLRM,
the user specifies

* the data `A`
* the observed entries `obs`
* the array of loss functions `losses`
* the regularizers `r` (which acts on `X`) and `rt` (which acts on `Y`)
* the rank `k`

Losses and regularizers must be of type `Loss` and `Regularizer`, respectively,
and may be chosen from a list of supported losses and regularizers, which include

* quadratic loss `quadratic`
* hinge loss `hinge`
* l1 loss `l1`
* ordinal hinge loss `ordinal_hinge`
* quadratic regularization `quadreg`
* no regularization `zeroreg`
* nonnegative constraint `nonnegative` (eg, for nonnegative matrix factorization)
* 1-sparse constraint `onesparse` (eg, for k-means)

Users may also implement their own losses and regularizers; 
see `loss_and_reg.jl` for more details.

For example, the following code forms a k-means model with $k=5$ on the matrix `A`:

	m,n,k = 100,100,5
	Y = randn(k,n)
	A = zeros(m,n)
	for i=1:m
		A[i,:] = Y[mod(i,k)+1,:]
	end
	losses = fill(quadratic(),n)
	rt = zeroreg()
	r = onesparse() 
	glrm = GLRM(A,losses,rt,r,k)

For more examples, see `examples/simple_glrms.jl`.

To fit the model, we call

	X,Y,ch = autoencode!(glrm)

which runs an alternating directions proximal gradient method on `glrm` to find the 
`X` and `Y` minimizing the objective function.
(`ch` gives the convergence history; see [Technical details](https://github.com/madeleineudell/GLRM.jl#technical-details) below for more information.)
The fields `glrm.X` and `glrm.Y` are also set by this call.

# Scaling and offsets

# Fitting DataFrames

# Technical details

## Optimization

### Warm start

### Parameters

### Convergence
`ch` gives the convergence history so that the success of the optimization can be monitored;
`ch.objective` stores the objective values, and `ch.times` captures the times these objective values were achieved.