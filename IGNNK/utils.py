import numpy as np
import csv
import os
from torch import nn


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        f.__next__()
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]
        distances = np.array([i[2] for i in edges])
    std = distances.std()
    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    for i, j, dist in edges:
        A[i, j] = np.exp(-np.square(dist / std))
    
    noise = np.random.rand(num_of_vertices,num_of_vertices)
    A = np.where((noise<0.2) & (A==0),np.exp(-np.square(1 / std)),A)
    return A


def random_walk_normalize(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_random_walk: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.sum(W, axis=1)
    D[D == 0] = 1e-10
    A_wave = np.multiply(np.reciprocal(D).reshape((-1, 1)), W)
    return A_wave


def load_pems_data(true_datapath,miss_datapath, distance_df_filename, num_of_vertices=170):
    miss = np.load(miss_datapath)
    mask = miss['mask'].transpose(1, 0, 2)[:, :, 0]
    miss_data = miss['data'].transpose(1, 0, 2).astype(np.float32)[:, :, 0]
    miss_data[np.isnan(miss_data)] = 0

    true_data = np.load(true_datapath)['data'].transpose(1, 0, 2).astype(np.float32)[:, :, 0]
    true_data[np.isnan(true_data)] = 0
    
    A = get_adjacency_matrix(distance_df_filename, num_of_vertices)

    #A = A +np.random.random(size=A.shape).astype(A.dtype)
    #miss_data = miss_data +np.random.normal(size=miss_data.shape,loc=10,scale=10).astype(miss_data.dtype)

    A_q = random_walk_normalize(A)
    A_h = random_walk_normalize(A.T)
    return miss_data, true_data, mask, A_q, A_h


class MaskL2Loss(nn.Module):
    def __init__(self):
        super(MaskL2Loss, self).__init__()

    def forward(self, pred, target, mask):
        assert mask.max() <= 1 + 1e-6
        diff = (pred - target) ** 2
        diff = diff * mask
        return diff.sum() / (mask.sum()+1)
