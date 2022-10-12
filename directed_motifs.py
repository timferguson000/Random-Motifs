# WORK IN PROGRESS!!!

# libraries

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm



# n = number of vertices
# m = number of possible directed edges

n = 25
m = n*(n-1)

# p = probability of an edge being included
# t = threshold value for including edge

p = 0.001
t = norm.ppf(p)

# a_id = 1 (this is a choice for normalization and is the result of our choice for t)
# p^2(1 + a_recip) = probability of the reciprocal motif
# p^2(1 + a_conv) = probability of the convergent motif
# p^2(1 + a_div) = probability of the divergent motif
# p^2(1 + a_chain) = probability of the chain motif
# p^2(1 + a_disj) = probability of the disjoint motif (this is technically
# not a motif but is included to complete the coherent configuration)

a_id = 1
a_recip = 0
a_div = 0
a_chain = 0
a_conv = 0
a_disj = 0
