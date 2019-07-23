# To Do

* Set up a single command for multiple imputation:
	a) pick m random subsets of data,
	b) choose model of rank k, regularization constant α for each subset,
	c) impute missing data from each of the m selected models.
	* (nandana will do this)

* Documentation!
	* how to think about mpca
	* imputation
	* error metrics for cross validation
	* new syntax for fitting data frame
		* how to specify loss function(s)
	* parallel fitting
	* full rank model / prisma

* Poisson loss
	* scaling?
	* to log or not to log? that is the interpretative issue

# Bugs

* init_nndsvd! doesn't work (probably an upgrade-to-1.0 bug)
* M_estimator doesn't work (losses.jl); bug in Optim?
* sample doesn't work
* lots of bugs in fit_dataframe_w_type_imputation; deprecated for now. (also it's an odd thing to do.)
* imputation doesn't return correct type (for dataframes)

# How to register/publish a new version of the package

1. update version number in Project.toml
2. navigate to commit that you want tagged on github
3. comment @Registrator register
4. monitor resulting PR on the general registry to see if any bugs are found
5. when PR is accepted, use Tagger to make github release
