# libraries
    
import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
import networkx as nx

# inputs
# n = number of vertices
# p = probability of an edge being included
# p^2(1 + a_conn) = probability of the connected motif
# p^2(1 + a_disj) = probability of the disjoint motif (this is technically
# not a motif but is included to complete the coherent configuration)

# global variables
# R_conn = indicator matrix for connected motif

# parameters: n (int), disp (optional boolean)
# returns: list of integer matrices [rho_id, rho_conn, rho_disj]; matrices computed for n
# optional: if disp=True (default disp=False), displays rho_id, rho_conn, and rho_disj

def undirected_rho_list(n, disp=False):

  # rho matrices
  rho_id = np.identity(3).astype(int)

  rho_conn = np.array([[0,1,0],
                       [2*n-4,n-2,4],
                       [0,n-3,2*n-8]]).astype(int)

  rho_disj = np.array([[0,0,1],
                       [0,n-3,2*n-8],
                       [(n-2)*(n-3)/2,(n-3)*(n-4)/2,(n-4)*(n-5)/2]]).astype(int)

  # display rho matrices if disp=True
  if disp:
    print(\"rho_id \\n{} \\n \\nrho_conn \\n{} \\n \\nrho_disj \\n{}\".format(rho_id, rho_conn, rho_disj))

  # returns list of rho matrices
  return [rho_id, rho_conn, rho_disj]

# example code

undirected_rho_list(6,disp=True)
          
# parameters: n (int), a_conn (float), a_disj (float)
# returns: if square root exists, list of three floats b; coefficients
# of positive definite square root, otherwise "Square root does not exist"

def square_root_coefficients(n, a_conn, a_disj):

  # a_id = 1 normalization
  a_id = 1

  # lists of coefficients and rho matrices
  a = [a_id, a_conn, a_disj]
  rho_list = undirected_rho_list(n)
          
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
  gram = np.zeros((3,3))
  for i in range(3):
    for j in range(3):
      gram[i,j] = np.trace(np.transpose(rho_list[i])@rho_list[j])

  vec = np.zeros(3)
  for i in range(3):
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
# returns: k (int); labeled undirected edge connecting nodes i and j

def edge(i,j):
  if i > j:
    return int(i*(i-1)/2 + j)\n",
  else:
    return int(j*(j-1)/2 + i)\n",

# parameters: k (int)
# returns: i (int), j (int); labeled nodes connected by undirected edge k with i > j

def edge_inv(k):
  i = 1
  while (i+1)*i/2 <= k:
    i += 1
  j = int(k - i*(i-1)/2)
  return i, j
          
# parameters: n (int)
# returns: nothing; constructs glogal variable R_conn

def undirected_row_col_conn(n):

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
  
undirected_row_col_conn(6)

print(R_conn)
          
# parameters: n (int), p (float), a_conn (float), a_disj (float)
# returns: edges for random graph
# note: n must match n used for global variables row and col from undirected_row_col_conn

def undirected_random_graph(n, p, a_conn, a_disj, disp_iid=False, disp_corr=False):

  # m = number of possible undirected edges
  m = int(n*(n-1)/2)

  # vector of independent standard normal random variables for each possible edge
  x_iid = np.random.normal(0,1,m)

  # displays x_iid if disp_iid=True
  if disp_iid:
    print(\"x_iid \\n{}\".format(x_iid))

  # computes coefficients of positive definite square root
  [b_id, b_conn, b_disj] = square_root_coefficients(n, a_conn, a_disj)

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
  
undirected_row_col_conn(6)

print(R_conn)

undirected_random_graph(6, 0.1, 0, 0, disp_iid=True, disp_corr=True)
          
# parameters: n (int), edges (int list)
# returns: adjacency matrix for edges

def undirected_adj(n, edges):

  adj = np.zeros((n,n)).astype(int)

  k = 0
  while k < len(edges):
    if edges[k] == 1:
      i, j = edge_inv(k)
      adj[i,j] = adj[j,i] = 1
  k += 1

  return adj
          
# example code
# optional: a_disj = -4*a_conn/(n-3) to make 2*(n-2)*a_conn + ((n-2)*(n-3)/2)*a_disj = 0

n = 10

# create global variable R_conn for given n

undirected_row_col_conn(n)

# for given n generate random graphs for any value of p and any values of
# a_conn and a_disj for which square root exists

edges = undirected_random_graph(n,0.25,0,0)
adj = undirected_adj(n, edges)
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
