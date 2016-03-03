# To Do

* Implementation of max-norm regularization.
	* check convergence
	* check converges with more interesting losses
	
* Make multidimensional categorical hinge loss

* Set up a single command to: 
	a) pick m random subsets of data, 
	b) choose model of rank k, regularization constant α for each subset, 
	c) impute missing data from each of the m selected models.
	* (nandana will do this)

* Call julia from R
	* i write a wrapper
	* nandana writes the UX

* Documentation!
	* functional forms and names of all loss functions
	* how to think about mpca
	* scaling
		* prob scale
	* imputation
	* error metrics for cross validation
	* new syntax for fitting data frame
		* how to specify loss function(s)
	* parallel fitting
	* full rank model / prisma

* Poisson loss
	* scaling?
	* to log or not to log? that is the interpretative issue