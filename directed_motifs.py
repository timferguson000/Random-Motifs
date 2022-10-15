WORK IN PROGRESS!!!

# libraries
    
import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
import networkx as nx

# inputs
# n = number of vertices
# p = probability of an edge being included
# p^2(1 + a_recip) = probability of the reciprocal motif
# p^2(1 + a_div) = probability of the divergent motif
# p^2(1 + a_chain) = probability of the chain motif
# p^2(1 + a_anti) = probability of the anti chain motif
# p^2(1 + a_conv) = probability of the convergent motif
# p^2(1 + a_disj) = probability of the disjoint motif (this is technically
# not a motif but is included to complete the coherent configuration)

# global variables
# R_conv = indicator matrix for convergent motif
# R_chain = indicator matrix for chain motif
# R_anti = indicator matrix for anti chain motif
# R_div = indicator matrix for divergent motif

# parameters: n (int), disp (optional boolean)
# returns: list of integer matrices [rho_id, rho_recip, rho_div, rho_chain, rho_anti, rho_conv, rho_disj]; matrices computed for n
# optional: if disp=True (default disp=False), displays rho_id, rho_recip, rho_div, rho_chain, rho_anti, rho_conv, and rho_disj

def directed_rho_list(n, disp=False):

  # rho matrices
  
  rho_id = np.identity(7)

  rho_recip = np.array([[0,1,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,1]])

  rho_div = np.array([[0,0,1,0,0,0,0],
                       [0,0,0,0,1,0,0],
                       [n-2,0,n-3,0,0,0,0],
                       [0,0,0,0,0,1,1],
                       [0,n-2,0,0,n-3,0,0],
                       [0,0,0,1,0,0,0],
                       [0,0,0,n-3,0,n-3,n-4]])

  rho_chain = np.array([[0,0,0,1,0,0,0],
                         [0,0,0,0,0,1,0],
                         [0,n-2,0,n-3,0,0,0],
                         [0,0,0,0,1,0,1],
                         [n-2,0,0,0,0,n-3,0],
                         [0,0,1,0,0,0,1],
                         [0,0,n-3,0,n-3,0,n-4]])

  rho_anti = np.array([[0,0,0,0,1,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,0,0,0,1,1],
                        [n-2,0,n-3,0,0,0,0],
                        [0,0,0,1,0,0,1],
                        [0,n-2,0,0,n-3,0,0],
                        [0,0,0,n-3,0,n-3,n-4]])

  rho_conv = np.array([[0,0,0,0,0,1,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,1],
                        [0,n-2,0,n-3,0,0,0],
                        [0,0,1,0,0,0,1],
                        [n-2,0,0,0,0,n-3,0],
                        [0,0,n-3,0,n-3,0,n-4]])

  rho_disj = np.array([[0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,n-3,n-3,n-4],
                        [0,0,0,0,n-3,n-3,n-4],
                        [0,0,n-3,n-3,0,0,n-4],
                        [0,0,n-3,n-3,0,0,n-4],
                        [(n-3)*(n-2),(n-3)*(n-2),(n-4)*(n-3),(n-4)*(n-3),(n-4)*(n-3),(n-4)*(n-3),(n-5)*(n-4)]])
  
  # display rho matrices if disp=True
  if disp:
    print("rho_id \\n{}
           \\n \\nrho_recip \\n{}
           \\n \\nrho_div \\n{}
           \\n \\nrho_chain \\n{}
           \\n \\nrho_anti \\n{}
           \\n \\nrho_conv \\n{}
           \\n \\nrho_disj \\n{}".format(rho_id, rho_recip, rho_div, rho_chain, rho_anti, rho_conv, rho_disj))

  # returns list of rho matrices
  return [rho_id, rho_recip, rho_div, rho_chain, rho_anti, rho_conv, rho_disj]

# example code

directed_rho_list(6,disp=True)
          
# parameters: n (int), a_conn (float), a_disj (float)
# returns: if square root exists, list of three floats b; coefficients
# of positive definite square root, otherwise "Square root does not exist"

def square_root_coefficients(n, a_recip, a_div, a_chain, a_anti, a_conv, a_disj):

  # a_id = 1 normalization
  a_id = 1

  # lists of coefficients and rho matrices
  a = [a_id, a_recip, a_div, a_chain, a_anti, a_conv, a_disj]
  rho_list = directed_rho_list(n)
          
  # linear combination
  rho = np.dot(a, rho_list)\n",

  # positive definite square root; returns "Square root does not exist" if
  # square root does not exist
  d, p = np.linalg.eig(rho)
  if sum(d < 0) > 0:
    return "Square root does not exist"
  d_root = np.diag(np.sqrt(d))
  rho_root = p@d_root@np.linalg.inv(p)

  # constructs gram matrix and inhomogeneous part
  gram = np.zeros((7,7))
  for i in range(7):
    for j in range(7):
      gram[i,j] = np.trace(np.transpose(rho_list[i])@rho_list[j])

  vec = np.zeros(7)
  for i in range(7):
    vec[i] = np.trace(np.transpose(rho_root)@rho_list[i])

  # solves linear system
  b = np.linalg.solve(gram, vec)

  # returns coefficients
  return b
          
# example code

print(square_root_coefficients(10, 0, 0))
print(square_root_coefficients(10, 0.1, 0))
print(square_root_coefficients(10, 1, 0))

          
          
          
          
          
          
          
          
          
          
          
          
          
# implements edge ordering e_0 = 1 <-> 0, e_1 = 2 <-> 0, e_2 = 2 <-> 1,
# e_3 = 3 <-> 0, e_4 = 3 <-> 1, ..., e_{n*(n-1)/2-1} = n-1 <-> n-2
# edge e_k = i <-> j where k = max*(max-1)/2 + min with max = max(i,j) and min = min(i,j)
          
          
          
          

# parameters: i (int), j (int)
# returns: k (int); labeled directed edge from node i to node j

def directed_edge(i,j):
  
          
          
          

# parameters: k (int)
# returns: i (int), j (int); i tail and j head of directed edge k

def directed_edge_inv(k):
  
          
          
          
          
# parameters: n (int)
# returns: nothing; constructs global variables R_recip, R_div, R_chain, R_anti, R_conv

def directed_row_col_conn(n):

          
          
          
          
          
# initialize arrays for length
data = row = col = np.ones(n*(n-1)*(n-2)).astype(int)
index = 0

for i in range(0,n):
  for j in range(0,i):
    for k in range(0,j):

    # three undirected edges connecting three nodes
    e_1 = edge(i,j)\n",
    e_2 = edge(j,k)\n",
    e_3 = edge(k,i)\n",

    # non-zero entries for connected motif
    row[index] = e_1
    col[index] = e_2
    index += 1

    row[index] = e_2
    col[index] = e_1
    index += 1

    row[index] = e_2
    col[index] = e_3
    index += 1

    row[index] = e_3
    col[index] = e_2
    index += 1

    row[index] = e_3
    col[index] = e_1
    index += 1

    row[index] = e_1
    col[index] = e_3
    index += 1

    # m = number of possible undirected edges
    m = int(n*(n-1)/2)

    # global variable R_conn
    global R_conn

    # constructs sparse matrix R_conn
    R_conn = csr_matrix((data, (row, col)), shape = (m, m)).toarray()

    # returns nothing
    return  

          
          
          
          
          
          
          
          
# example code
  
directed_row_col_conn(6)

print(R_recip)
print(R_div)
print(R_chain)
print(R_anti)
print(R_conv)
          
# parameters: n (int), p (float), a_conn (float), a_disj (float)
# returns: edges for random graph
# note: n must match n used for global variables row and col from undirected_row_col_conn

def directed_random_graph(n, p, a_recip, a_div, a_chain, a_anti, a_conv, a_disj, disp_iid=False, disp_corr=False):

  # m = number of possible undirected edges
  m = n*(n-1)

  # vector of independent standard normal random variables for each possible edge
  x_iid = np.random.normal(0,1,m)

  # displays x_iid if disp_iid=True
  if disp_iid:
    print(\"x_iid \\n{}\".format(x_iid))

  # computes coefficients of positive definite square root
  [b_id, b_recip, b_div, b_chain, b_anti, b_conv, b_disj] = square_root_coefficients(n, a_conn, a_disj)

  # vector of correlated standard normal random variables for each possible edge
  x_corr = (b_id-b_disj)*x_iid + (b_conn-b_disj)*R_conn@x_iid + b_disj*np.sum(x_iid)*np.ones(m)

  # displays x_corr if disp_corr=True
  if disp_corr:
    print(\"x_corr \\n{}\".format(x_corr))

  # t = threshold value for including edge
  t = norm.ppf(p)

  # includes edges based on x_corr and the threshold t
  edges = [1 if element < t else 0 for element in x_corr]

  # returns edges
  return edges
  
# example code
  
directed_row_col_conn(6)

print(R_recip)
print(R_div)
print(R_chain)
print(R_anti)
print(R_conv)

directed_random_graph(6, 0.1, 0, 0, disp_iid=True, disp_corr=True)
          
# parameters: n (int), edges (int list)
# returns: adjacency matrix for edges

def directed_adj(n, edges):

  adj = np.zeros((n,n)).astype(int)

  k = 0
  while k < len(edges):
    if edges[k] == 1:
      i, j = directed_edge_inv(k)
      adj[i,j] = adj[j,i] = 1
  k += 1

  return adj
          
# example code
# optional: a_disj = -4*a_conn/(n-3) to make 2*(n-2)*a_conn + ((n-2)*(n-3)/2)*a_disj = 0????????????

n = 10

# create global variables R_recip, R_div, R_chain, R_anti, R_conv for given n

directed_row_col_conn(n)

# for given n generate random graphs for any value of p and any values of
# a_recip, a_div, a_chain, a_anti, a_conv, and a_disj for which square root exists

edges = undirected_random_graph(n,0.25,0,0,0,0,0,0)
adj = directed_adj(n, edges)
          
          
          
          
          
graph = nx.from_numpy_matrix(adj)

print(\"edges \\n{} \\n \\nadj \\n{} \\n\".format(edges, adj))
nx.draw(graph)
      
# example code
# optional: a_disj = -4*a_conn/(n-3) to make 2*(n-2)*a_conn + ((n-2)*(n-3)/2)*a_disj = 0

n = 100

# create global variable R_conn for given n

undirected_row_col_conn(n)

# for given n generate random graphs for any value of p and any values of
# a_conn and a_disj for which square root exists

edges = undirected_random_graph(n,0.1,0,0)
adj = undirected_adj(n, edges)
graph = nx.from_numpy_matrix(adj)

nx.draw(graph)
      
# example code

edges = undirected_random_graph(n,0.1,0.00001,0)
adj = undirected_adj(n, edges)
graph = nx.from_numpy_matrix(adj)

nx.draw(graph)
      
# example code

edges = undirected_random_graph(n,0.05,0,0)
adj = undirected_adj(n, edges)
graph = nx.from_numpy_matrix(adj)

nx.draw(graph)
      
# example code

edges = undirected_random_graph(n,0.05,0.00001,0)
adj = undirected_adj(n, edges)
graph = nx.from_numpy_matrix(adj)

nx.draw(graph)



















































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
