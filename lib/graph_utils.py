import numpy as np
import scipy.sparse as sp

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def loadGraph(args):
    adj = np.load(args.adj_file)
    adj = adj + np.eye(adj.shape[0])
    graphwave = get_eigv(adj, args.h*args.d)
    adjgat = np.load(args.adjgat_file)
    return adjgat, graphwave
