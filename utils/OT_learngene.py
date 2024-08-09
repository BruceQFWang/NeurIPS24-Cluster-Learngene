"""
We refer to FLOT (https://github.com/valeoai/FLOT/blob/master/flot/tools/ot.py)
when implementing the sinkhorn algorithm.
"""
# [[0.92, 0.92, 0.90],  [0.92, 0.91, 0.90], [0.81, 0.80, 0.79], [0.52, 0.48, 0.43], [0.61, 0.61, 0.58], [0.69, 0.69, 0.62]]

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def OT(C, epsilon=0.03, OT_iter=4):
    B, N1, N2 = C.shape

    # Entropic regularisation
    K = torch.exp( -C / epsilon)

    # Init. of Sinkhorn algorithm
    a = torch.ones((B, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob1 = torch.ones((B, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob2 = torch.ones((B, N2, 1), device=C.device, dtype=C.dtype) / N2

    # Sinkhorn algorithm
    for _ in range(OT_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = prob2 / (KTa + 1e-8)

        # Update a
        Kb = torch.bmm(K, b)
        a = prob1 / (Kb + 1e-8)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    return T

def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization  
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    # Avoiding poor math condition
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)

def OT_module(OT_iter=4):

    # Matching cost
    '''Cost_dist, Support = Cost_Gaussian_function(pc1, pc2)
    Cost_norm = Cost_cosine_distance(norm1, norm2)
    Cost = Cost_dist + Cost_norm

    if color1 is not None:
        Cost_color, _ = Cost_Gaussian_function(color1, color2, theta_2=0.12)
        Cost = Cost + Cost_color'''
    
    similarity_score = torch.tensor([[0.92, 0.92, 0.90],  [0.92, 0.91, 0.90], [0.81, 0.80, 0.79], [0.52, 0.48, 0.43], [0.61, 0.61, 0.58], [0.69, 0.69, 0.62]]) 
    
    Cost = 1.0 - similarity_score #改成score   
    
    # Optimal transport plan
    T = OT(Cost, epsilon=0.03, OT_iter=OT_iter)

    # Hard correspondence matrix
    indices = T.max(2).indices
    matrix2 = torch.nn.functional.one_hot(indices, num_classes=T.shape[2])

    # Remove some invalid correspondences with large displacements
    valid_map = matrix2 * Support
    valid_vector = torch.sum(valid_map, 2)

    return indices, valid_vector.float()

# indices, valid_vector = OT_module()

layer_similarity = torch.tensor([[0.92, 0.92, 0.90],  [0.92, 0.91, 0.90], [0.81, 0.80, 0.79], [0.52, 0.48, 0.43], [0.61, 0.61, 0.58], [0.69, 0.69, 0.62]]).to(device) 
Cost = 1.0 - layer_similarity 
NA = 6 # number of layers in the Ancestry model
ND = 3 # number of layers in the Descendant model
r = torch.ones((NA, 1), device=Cost.device, dtype=Cost.dtype) / NA
c = torch.ones((ND, 1), device=Cost.device, dtype=Cost.dtype) / ND
lam = 10
P, d = compute_optimal_transport(M, r, c, lam=lam)
