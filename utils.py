import numpy as np
import torch

class Graph:
    def __init__(self):
        # MediaPipe Pose Landmarker connections (33 keypoints)
        self.num_node = 33
        self.edges = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20),
            (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
            (30, 32), (27, 31), (28, 32)
        ]
        self.adjacency = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        return A

def normalize_adjacency(A):
    # Add self-loops
    A_loop = A + np.eye(A.shape[0])
    # Compute degree matrix D
    row_sum = np.sum(A_loop, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    d_inv_sqrt[row_sum == 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    return np.dot(np.dot(D_inv_sqrt, A_loop), D_inv_sqrt)
