from numpy.random import rand, randn, choice
from numpy import hstack

# from LowRankModels import * # import julia functions into python
import julia
j = julia.Julia()
j.using("LowRankModels")

# from julia import LowRankModels as lrm

# loss types to test - all these are Julia functions imported from LowRankModels
real_loss_types = [j.QuadLoss, j.HuberLoss]
bool_loss_types = [j.HingeLoss]
ordinal_loss_types = [j.OrdinalHingeLoss, j.BvSLoss]
categorical_loss_types = [j.MultinomialLoss, j.OvALoss]

#instantiate losses
ncat = 4 # maximum categorical levels
nord = 5 # maximum ordinal levels
real_losses = [l() for l in real_loss_types] # a python list of julia objects formed by calling julia functions
bool_losses = [l() for l in bool_loss_types]
ordinal_losses = [l(choice(range(3,nord))) for l in ordinal_loss_types]
categorical_losses = [l(choice(range(3,ncat))) for l in categorical_loss_types]
losses = real_losses + bool_losses + ordinal_losses + categorical_losses # concatenate python lists of julia objects

# scale losses for different columns
for loss in losses:
  j.scale_b(loss, rand()) # scale_b is the julia function scale!

# regularizers to test
regularizers = [j.QuadReg(), j.OneReg(5), j.NonNegConstraint(), j.KSparseConstraint(2)] # python list of julia objects

m = len(regularizers)
n = len(losses)

# matrices of data corresponding to each loss type
A_real = rand(m, len(real_losses))
A_bool = choice([True, False], size = (m, len(bool_losses)))
A_ord = choice(range(1,6), size = (m, len(ordinal_losses))) # use 1 indexing
A_cat = choice(range(1,4), size = (m, len(categorical_losses))) # use 1 indexing

# horizontally concatenate to form one big data matrix
A = hstack([A_real, A_bool, A_ord, A_cat])

# now call main function
# tricky business:
    # * A is a numpy array, will need to be cast as Julia array
    # * losses is a python list of julia objects, will need to be cast as a Julia Array{Loss,1}
    # * regularizers is a python list of julia objects, will need to be cast as a Julia Array{Regularizer,1}
    # * QuadReg() forms a julia object, no problem
    # * 2 is a python int, will need to be cast as a julia int
glrm = j.GLRM(A, losses, regularizers, j.QuadReg(), 2) # GLRM is a julia function from LowRankModels
j.fit_b(glrm) # fit_b is the julia function fit!
print("successfully fit matrix")

### now fit data frame
j.using("DataFrames")
# could also try a pandas table instead...?
A = j.DataFrame(A)
for i in range(10):
  A[choice(range(1,m+1)), choice(range(1,n+1))] = j.NA
obs = j.observations(A)
glrm = j.GLRM(A, losses, QuadReg(), QuadReg(), 2, obs=obs)
j.fit_b(glrm)
print("successfully fit dataframe")

# imputation and sampling
j.impute(glrm)
print("successfully imputed entries")
j.sample(glrm)
j.sample_missing(glrm)
print("successfully sampled from model")

### now fit sparse matrix
from scipy.sparse import random

m, n = 100, 200
A = random(m, n, .5, format='csc')
glrm = j.GLRM(A, QuadLoss(), QuadReg(), QuadReg(), 5)
j.fit_b(glrm)
print("successfully fit csc sparse GLRM")

A = random(m, n, .5, format='coo')
glrm = j.GLRM(A, QuadLoss(), QuadReg(), QuadReg(), 5)
j.fit_b(glrm)
print("successfully fit coo sparse GLRM")

A = random(m, n, .5, format='csr')
glrm = j.GLRM(A, QuadLoss(), QuadReg(), QuadReg(), 5)
j.fit_b(glrm)
print("successfully fit csr sparse GLRM")

### the hardest test: what if someone passes a pandas dataframe?
import pandas as pd
import numpy as np
df2 = pd.DataFrame({ 'A' : 1.,
                      'B' : pd.Timestamp('20130102'),
                      'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                      'D' : np.array([3] * 4,dtype='int32'),
                      'E' : pd.Categorical(["test","train","test","train"]),
                      'F' : 'foo' })

# todo: finish writing this test and fit a GLRM on the pandas dataframe!
