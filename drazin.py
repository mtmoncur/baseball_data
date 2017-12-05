# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
import csv


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.allclose(la.det(A),0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        bool: True of Ad is the Drazin inverse of A, False otherwise.
    """
    ma_pow = np.linalg.matrix_power
    if not np.allclose(A.dot(Ad), Ad.dot(A)):
        return False
    if not np.allclose(ma_pow(A,k+1).dot(Ad), ma_pow(A,k)):
        return False
    if not np.allclose(np.dot(Ad,A.dot(Ad)), Ad):
        return False
    return True


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        Ad ((n,n) ndarray): The Drazin inverse of A.
    """
    n = A.shape[0]
    f_gre = lambda x: abs(x) > tol
    f_leq = lambda x: abs(x) <= tol
    Q1,S,k1 = la.schur(A, sort=f_gre)
    Q2,T,k2 = la.schur(A, sort=f_leq)
    U = np.hstack([S[:,:k1], T[:,:n-k1]])
    U_inv = la.inv(U)
    V = U_inv.dot(np.dot(A,U))
    Z = np.zeros_like(A).astype(np.float64)
    if k1 != 0:
        M_inv = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv
    return U.dot(np.dot(Z,U_inv))

# Problem 3
def effective_res(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ER ((n,n) ndarray): A matrix of which the ijth entry is the effective
        resistance from node i to node j.
    """
    n = A.shape[0]
    L = (np.diag(np.sum(A, axis=1)) - A).astype(np.float)
    R = np.zeros_like(L)
    for j in range(n):
        print('{} of {}'.format(j,n))
        L_prime = L.copy()
        L_prime[j] = np.zeros(n)
        L_prime[j,j] = 1
        Lpd = drazin_inverse(L_prime)
        R[:,j] = Lpd.diagonal()
        R[j,j] = 0
    return R
        


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.
        
        Parameters:
            filename (str): The name of a file containing graph data.
        """
        '''s = self
        s.names_dct = {}
        with open(filename, 'r') as target:
            names_set = set()
            content = csv.reader(target)
            connections = []
            for i in content:
                names_set.add(i[0])
                names_set.add(i[1])
                connections.append(i)
            s.names = list(names_set)
            for i, name in enumerate(s.names):
                s.names_dct[name] = i
            n = len(names_set)
            s.A = np.zeros((n,n))
            for a, b in connections:
                a_ind = s.names_dct[a]
                b_ind = s.names_dct[b]
                s.A[a_ind,b_ind] = 1
                s.A[b_ind,a_ind] = 1
        target.close()
        s.res = effective_res(s.A)
        '''
        pass
        


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.
        
        Parameters:
            node (str): The name of a node in the network.
        
        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.
        
        Raises:
            ValueError: If node is not in the graph.
        """
        if node is not None:
            if node not in self.names:
                raise ValueError("Given node not found for predict_link")
            ind = self.names_dct[node]
            temp = (1-self.A[ind])*self.res[ind]
            temp[temp==0] = 1e10
            best_node = np.argmin(temp)

            return self.names[best_node]
            
        mask = (self.A == 0)
        di = np.diag_indices(len(self.A))
        mask[di] = False
        min_val = np.min(self.res[mask])
        
        loc = np.where(self.res == min_val)
        ind1 = loc[0][0]
        ind2 = loc[1][0]
        
        node1 =  self.names[ind1]
        node2 = self.names[ind2]
        return node1, node2
            
        

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if (node1 not in self.names) or (node2 not in self.names):
            raise ValueError("Given node not found for add_link")
        ind1 = self.names_dct[node1]
        ind2 = self.names_dct[node2]
        self.A[ind1, ind2] = 1
        self.A[ind2, ind1] = 1
        self.res = effective_res(self.A)

        
