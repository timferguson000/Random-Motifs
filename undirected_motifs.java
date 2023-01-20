/*
    This is a Java implementation of the algorithm published in the article
    "Motifs, coherent configurations and second order network generation"
    in the journal Physica D: Nonlinear Phenomena by Jared Bronski and Tim Ferguson.
    The algorithm generalizes Erdos-Renyi random graphs be introducing correlation between
    the random variables used to determine if an edge is included or not. This algorithm is
    based on original insights into the structure of these graphs and uses an algebraic
    object called a coherent configuration to exactly replicate the desired distribution with
    high efficiency.
    
    Complexity analysis (let n denote the number of vertices in the random graph):
    The algorithm requires O(n^2) storage which is best possible since graphs from the desired distribution are dense (i.e. have order n^2 edges).
    The algorithm requries O(n^3) calculations which is the same complexity as the motifs of a graph with n vertices. Note that these calculations occur
    in three main stages the first of which is O(1) while the second two are O(n^2) and O(n^3) but can be completed with a parallel architecture.

    Journal publication: https://www.sciencedirect.com/science/article/abs/pii/S0167278921002657
    arXiv preprint: https://arxiv.org/abs/1808.05076
    Medium post: https://medium.com/@timferguson000/efficient-generation-of-erd%C3%B6s-r%C3%A9nyi-random-graphs-with-prescribed-motif-distributions-bd4581134447
 */

package undirectedmotifs;

import java.lang.Math;
import Jama.*; // documentation https://math.nist.gov/javanumerics/jama/doc/Jama/Matrix.html
import java.util.Random; 
import org.jgrapht.*; // documentation https://jgrapht.org/guide/UserOverview#hello-jgrapht
import org.jgrapht.graph.*;

public class UndirectedMotifs {
    
    /*
       Given parameter n:
       Construct rho matrices
    */
    
    public static Matrix[] rhoMatrices(int n) {
        
        double[][] rho_0 = {
                            {1,0,0},
                            {0,1,0},
                            {0,0,1}
                           };       
        double[][] rho_1 = {
                            {0,1,0},
                            {2*n-4,n-2,4},
                            {0,n-3,2*n-8}
                           };
        double[][] rho_2 = {
                            {0,0,1},
                            {0,n-3,2*n-8},
                            {(n-2)*(n-3)/2,(n-3)*(n-4)/2,(n-4)*(n-5)/2}
                           };
        
        return new Matrix[]{
                            new Matrix(rho_0,3,3),
                            new Matrix(rho_1,3,3),
                            new Matrix(rho_2,3,3)
                           };
    }
    
    /*
       Given parameters n, a_0, a_1, a_2:
       Determine if the covariance matrix is positive definite
    
       If covariance matrix is positive definite:
       Compute coefficients of the positive definite square root of rho
       Set indicator to 1.0
       Print that parameters are admissible and print parameter values
    
       If covariance matrix is no positive definite:
       Keep default indicator value 0.0
       Print that parameters are not admissible and print parameter values
       
    */
    
    public static double[] coefficientsSqrt(int n, double a_0, double a_1, double a_2) {
        
        /*
           Default values of coefficient and indicator variables
        */
        
        double b_0 = 0.0;
        double b_1 = 0.0;
        double b_2 = 0.0;
        double indicator = 0.0;
        
        /*
           Constructs matrix part of matrix equation
        */
        
        Matrix[] rho_matrices = rhoMatrices(n);
        double[][] a = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                a[i][j] = ((rho_matrices[i].transpose()).times(rho_matrices[j])).trace();
            }
        }
        Matrix A = new Matrix(a,3,3);

        Matrix rho = (rho_matrices[0].times(a_0)).plusEquals(rho_matrices[1].times(a_1)).plusEquals(rho_matrices[2].times(a_2));  
        Matrix D = rho.eig().getD();
        double d_0 = D.get(0,0);
        double d_1 = D.get(1,1);
        double d_2 = D.get(2,2);
        
        /*
           Test if covariance matrix is positive definite by checking if eigenvalues are positive
        */
        
        if ((d_0 > 0) && (d_1 > 0) && (d_2 > 0)) {
            
            indicator = 1.0; // Sets indicator to 1.0
            
            double d_0_sqrt = Math.sqrt(d_0);
            double d_1_sqrt = Math.sqrt(d_1);
            double d_2_sqrt = Math.sqrt(d_2);
            Matrix D_sqrt = new Matrix(new double[][]{
                                                      {d_0_sqrt,0,0},
                                                      {0,d_1_sqrt,0},
                                                      {0,0,d_2_sqrt}
                                                     },3,3); // Computes square root of diagonal eigenvalue matrix
            Matrix V = rho.eig().getV();
            Matrix rho_sqrt = V.times(D_sqrt.times(V.transpose())); // Constructs positive definite square root of rho
            double[] y = new double[3];
            for (int i = 0; i < 3; i++) {
                y[i] = ((rho_matrices[i].transpose()).times(rho_sqrt)).trace();
            }
            Matrix Y = new Matrix(y,3); // Constructs vector part of matrix equation
            Matrix B = A.solve(Y); // Solves matrix equation
            b_0 = B.get(0,0);
            b_1 = B.get(1,0);
            b_2 = B.get(2,0);
        }
        
        /*
           Prints if the parameters are admissible or not
        */
        
        if (indicator == 1.0) {
            System.out.println("Covariance matrix is positive definite. Paramaters are admissible:");
        } else {
            System.out.println("Covariance matrix is not positive definite. Paramaters are not admissible:");  
        }
        
        /*
           Prints values of parameters
        */
        
        System.out.println("n = " + n);
        System.out.println("a_0 = " + a_0);
        System.out.println("a_1 = " + a_1);
        System.out.println("a_2 = " + a_2);
        
        return new double[]{b_0,b_1,b_2,indicator};
    }
    
    /*
       Sets parameters n, p, t, a_0, a_1, a_2
       Determines if the parameters n, a_0, a_1, a_2 are admissible
    
       If parameters are admissible:
       Uses coefficients of positive definite square root of rho to construct random graph from specified distribution
       Produces Graph object "graph"
       Produces adjacency matrix "adj" as array double[][]
    */
    
    public static void main(String[] args) {
        
        /*
           Parameters
        */
        
        int n = 100; // Number of vertices
        double p = 0.1; // Probability of adding edge
        double t = -1.28; // t is the quantile for p under the standard normal distribution
        double a_0 = 1.0; // Normalization
        double a_1 = 0.1; // p^2*(1+a_1) is covariance of connected motif
        double a_2 = -0.001; // p^2*(1+a_2) is covariance of disjoint motif
        
        /*
           Calls coefficientSqrt
        */
        
        double[] coefficientsqrt = coefficientsSqrt(n,a_0,a_1,a_2);
        double b_0 = coefficientsqrt[0];
        double b_1 = coefficientsqrt[1];
        double b_2 = coefficientsqrt[2];
        double indicator = coefficientsqrt[3];
        
        /*
           If parameters are admissible: Generate random graph as Graph object "graph" and adjacency matrix "adj" as array object double[][]
        */
        
        if (indicator == 1.0) {
            System.out.println("Additional parameters:"); // Print value of p
            System.out.println("p = " + p);
            /*
               Creates symmetric array of i.i.d. standard normal random variables (diagonal remains 0.0)
               Computes sum
            */
            
            double sum = 0;
            Random rand = new Random();
            double[][] random_matrix = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    random_matrix[i][j] = rand.nextFloat();
                    random_matrix[j][i] = random_matrix[i][j];
                    sum += random_matrix[i][j];
                }
            }
            
            /*
               Initializes adjacency matrix "adj" as array double[][]
               Initializes Graph object "graph" with n vertices
            */
            
            int[][] adj = new int[n][n];
            Graph graph = new DefaultUndirectedGraph(DefaultEdge.class);
            for (int i = 0; i < n; i++) {
                graph.addVertex(i);
            }
            
            /*
               Computes the contribution of random variables to the edge connecting vertices i and j and compares to threshold t
            */
            
            for (int i = 0; i < n; i++) {
                for(int j = 0; j < i; j++) {
                    double sum_conn = 0;
                    for (int k = 0; k < n; k++) {
                        if ((k != i) && (k != j)) {
                            sum_conn += random_matrix[i][k];
                            sum_conn += random_matrix[j][k];
                        }
                    }
                    if ((b_2 - b_0)*random_matrix[1][0] + (b_1 - b_2)*sum_conn + b_2*sum < t) {
                        adj[i][j] = 1; // Includes edge in adjacency matrix
                        adj[j][i] = 1; // Adjacency matrix is symmetric since graph is undirected
                        graph.addEdge(i, j); // Included edge in graph
                    }
                }
            }
            
            /*
               Final result:
               Graph object "graph" (undirected graph) and double[][] object "adj" (adjacency matrix).
               Can use package jgrapht to display or apply graph algorithms to Graph object "graph".
            */
        }
    }   
}
