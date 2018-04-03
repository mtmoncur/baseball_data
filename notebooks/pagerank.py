# pagerank.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix as dok

# Problem 1
def to_matrix(filename, n):
    """Return the nxn adjacency matrix described by datafile.

    Parameters:
        datafile (str): The name of a .txt file describing a directed graph.
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile

    Returns:
        A SciPy sparse dok_matrix.
    """
    content = ''
    with open(filename, 'r') as target:
        content = target.readlines()
    target.close()

    A = dok((n,n), dtype = np.int64)
    for i, line in enumerate(content):
        try:
            i,j = map(int, line.strip().split())
            A[i, j] = 1
        except:
            pass

    return A

# Problem 2
def calculateK(A,N):
    """Compute the matrix K as described in the lab.

    Parameters:
        A (ndarray): adjacency matrix of an array
        N (int): the datasize of the array

    Returns:
        K (ndarray)
    """
    B = A.copy().astype(np.float64)
    
    zero = np.zeros(N)
    one = np.ones(N)

    for i, row in enumerate(B):
        if np.allclose(row, zero):
            B[i] = one
    d = np.sum(B, axis = 1)

    return (B).T/d


# Problem 3
def iter_solve(adj, N=None, d=.85, tol=1E-5):
    """Return the page ranks of the network described by 'adj'.
    Iterate through the PageRank algorithm until the error is less than 'tol'.

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    dif = 1
    if N is None:
        N = adj.shape[0]

    K = calculateK(adj[:N,:N], N)
    p = np.random.random(N)
    one = np.ones(N)

    while dif >= tol:
        pnew = d*K.dot(p) + ((1-d)/N)*one
        dif = np.linalg.norm(pnew - p)
        p = pnew

    return p

# Problem 4
def eig_solve(adj, N=None, d=.85):
    """Return the page ranks of the network described by 'adj'. Use SciPy's
    eigenvalue solver to calculate the steady state of the PageRank algorithm

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    if N is None:
        N = adj.shape[0]

    K = calculateK(adj[:N,:N], N)
    p = np.random.random(N)
    one = np.ones((N, N))

    B = d*K + ((1-d)/N*one)

    val, vect = spla.eig(B)
    p = np.real(vect[:, abs(val - 1) < 1e-10])

    #print "p shape:", p.shape
    return p.ravel()/np.sum(p)


# Problem 5
def team_rank(filename='ncaa2013.csv'):
    """Use iter_solve() to predict the rankings of the teams in the given
    dataset of games. The dataset should have two columns, representing
    winning and losing teams. Each row represents a game, with the winner on
    the left, loser on the right. Parse this data to create the adjacency
    matrix, and feed this into the solver to predict the team ranks.

    Parameters:
        filename (str): The name of the data file.
    Returns:
        ranks (list): The ranks of the teams from best to worst.
        teams (list): The names of the teams, also from best to worst.
    """
    team_set = set()
    games = []
    
    with open(filename, 'r') as ncaafile:
        ncaafile.readline() #reads and ignores the header line
        for line in ncaafile:
            teams = line.strip().split(',') #split on commas
            #print teams
            team_set.add(teams[0])
            team_set.add(teams[1])
            games.append(teams)
    ncaafile.close()

    team_list = sorted([i for i in team_set])
    team_dct = {}
    for i, team in enumerate(team_list):
        team_dct[team] = i

    N = len(team_set)
    A = np.zeros((N, N))

    for winner, loser in games:
        i = team_dct[loser]
        j = team_dct[winner]
        
        
        A[i, j] = 1

    #print "A :", A
    raw_rank = eig_solve(A, d=0.7)
    ranking = raw_rank.argsort().tolist()
    #print ranking
    #print team_list
    #reverse the indices so that the highest rank is 1, lowest is N
    #ranking = (*np.ones_like(ranking) - ranking

    ranks_by_team = [(team_list[ind]) for rank, ind in enumerate(ranking[::-1])]
    teams_ranked = [(ranking.index(team_dct[team])) for team in team_list]
    #ranks_by_team = sorted(ranks_by_team, key = lambda x: x[0])
    
    return sorted(raw_rank, reverse = True), ranks_by_team