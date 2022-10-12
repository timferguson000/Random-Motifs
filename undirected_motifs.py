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

n = 50
p = 0.1
a_conn = 0
a_disj = 0

# other parameters
# m = number of possible directed edges
# t = threshold value for including edge
# a_id = 1 normalization

m = int(n*(n-1)/2)
t = norm.ppf(p)
a_id = 1

# constructs rho matrices for coherent configuration

rho_id = np.identity(3)

rho_conn = np.array([[0,1,0],
                     [2*n-4,n-2,4],
                     [0,n-3,2^n-8]])

rho_disj = np.array([[0,0,1],[0,n-3,2*n-8],[(n-2)*(n-3)/2,(n-3)*(n-4)/2,(n-4)*(n-5)/2]])

# constucts rho matrix

rho = a_id*rho_id + a_conn*rho_conn + a_disj*rho_disj

# computes positive definite square root of rho

u, s, vh = np.linalg.svd(rho)

s_root = np.diag(np.sqrt(s))

rho_root = u@s_root@vh

# finds coefficients of positive definite square root with respect to coherent configuration

rho_list = [rho_id,rho_conn,rho_disj]

gram = np.zeros((3,3))
vec = np.zeros(3)

for i in range(3):
    for j in range(3):
        gram[i,j] = np.trace(np.transpose(rho_list[i])@rho_list[j])

for i in range(3):
    vec[i] = np.trace(np.transpose(rho_root)@rho_list[i])
    
[b_id,b_conn,b_disj] = np.linalg.solve(gram, vec)

# constructs R_conn with edge order e_0 = 1 <-> 0, e_1 = 2 <-> 0, e_2 = 2 <-> 1,
# e_3 = 3 <-> 0, e_4 = 3 <-> 1, ..., e_{n*(n-1)/2-1} = n-1 <-> n-2

def edge(i,j):
    if i > j:
        return int(i*(i-1)/2 + j)
    else:
        return int(j*(j-1)/2 + i)

data = row = col = np.ones(n*(n-1)*(n-2)).astype(int)
index = 0

for i in range(0,n):
    for j in range(0,i):
        for k in range(0,j):
            
            # edges
            e_1 = edge(i,j)
            e_2 = edge(j,k)
            e_3 = edge(k,i)
            
            # conn
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

R_conn = csr_matrix((data, (row, col)), 
                          shape = (m, m)).toarray()

# creates x_iid and computes product with positive definite square root of correlation matrix

x_iid = np.random.normal(0,1,m)

x_corr = (b_id-b_disj)*x_iid + (b_conn-b_disj)*R_conn@x_iid + b_disj*np.sum(x_iid)*np.ones(m)

# includes edges based on x_corr and threshold t

edges = [1 if element < t else 0 for element in x_corr]

# constructs adjacency matrix

def edge_inv(k):
    i = 1
    while (i+1)*i/2 <= k:
        i += 1
    j = int(k - i*(i-1)/2)
    return i, j

adj = np.zeros((n,n)).astype(int)

k = 0
while k < len(edges):
    if edges[k] == 1:
        i, j = edge_inv(k)
        adj[i,j] = adj[j,i] = 1
    k += 1
    
# creates and displays graph

graph = nx.from_numpy_matrix(adj)
nx.draw(graph)

# prints number of edges vs expected number of edges

print(sum(edges))
print(p*m)
