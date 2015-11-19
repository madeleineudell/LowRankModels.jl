# To Do

## mpca changes

* make updates efficient in proxgrad
	* don't form a matrix for the gradient before passing it to gemm. call axpy instead.
* make SVD initialization work for OrdisticLoss
	* maybe, encode levels as a number and replicate the y vector found across columns of Y?