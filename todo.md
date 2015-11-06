# To Do

## mpca changes

* make updates efficient in proxgrad
	* don't allocate excessive memory to store gradients
	* use gemm! directly
* make initialization work for mpca
	* without allocating too much memory / with tight types