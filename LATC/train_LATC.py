


import nni
import numpy as np


def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat, dim, mode):
    index = list()
    index.append(mode)
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order='F'), 0, mode)


# In[2]:


def svt_tnn(mat, tau, theta):
    """
    Singular Value Thresholding(svt)
    Truncated Nuclear Norm(tnn)
    
    Parameters: 
    mat  - numpy.array(m,n)
    tau  - truncation value
    theta  -  truncation factor
    
    Returns:
    Completed MAT  -  numpy.array(m,n)
    """
    
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices=0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[: theta] = 1
        mid[theta: idx] = (s[theta: idx] - tau) / s[theta: idx]
        return (u[:, : idx] @ np.diag(mid)) @ (u[:, : idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    idx = np.sum(s > tau)
    vec = s[: idx].copy()
    vec[theta: idx] = s[theta: idx] - tau
    return u[:, : idx] @ np.diag(vec) @ v[: idx, :]


def compute_mae(var, var_hat):
    return np.sum(np.abs(var - var_hat)) / var.shape[0]


def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


# In[4]:


def print_result(it, tol, var, var_hat):
    mae = compute_mae(var, var_hat)
    rmse = compute_rmse(var, var_hat)
    mape = compute_mape(var, var_hat)
    print('Iter: {}'.format(it))
    print('Imputation MAE: {:.6}'.format(mae))
    print('Imputation RMSE: {:.6}'.format(rmse))
    print('Imputation MAPE: {:.6}'.format(mape))
    print()
    if use_nni:
        nni.report_final_result(mae.item())


# How to create $\boldsymbol{\Psi}_{0},\boldsymbol{\Psi}_{1},\ldots,\boldsymbol{\Psi}_{d}$?

# In[5]:


from scipy import sparse
from scipy.sparse.linalg import spsolve as spsolve


def generate_Psi(dim_time, time_lags):
    """
    Generate the coefficient matrix for the autoregression
    
    Parameters: 
    dim_time  - 
    time_lags  - numpy.array(len) :: Autoregressive lag set
    
    Returns:
    Psis  -  numpy.array(time_lags + 1 , dim_time - max_lag , dim_time) # Psis[lag,T_a,T_b] -> 
    
    """
    Psis = []
    max_lag = np.max(time_lags)
    for i in range(len(time_lags) + 1):
        row = np.arange(0, dim_time - max_lag)
        if i == 0:
            col = np.arange(0, dim_time - max_lag) + max_lag
        else:
            col = np.arange(0, dim_time - max_lag) + max_lag - time_lags[i - 1]
        data = np.ones(dim_time - max_lag)
        Psi = sparse.coo_matrix((data, (row, col)), shape=(dim_time - max_lag, dim_time))
        Psis.append(Psi)
    return Psis


# In[6]:


import numpy as np

# Example
dim_time = 5
time_lags = np.array([1, 3])
Psis = generate_Psi(dim_time, time_lags)


def latc(dense_tensor, sparse_tensor, time_lags, alpha, rho0, lambda0, theta,
         epsilon=1e-4, maxiter=100, K=3):
    """
    Low-Rank Autoregressive Tensor Completion (LATC)
    
    Parameters: 
    dense_tensor  - numpy.array(Node, points_per_day, days) :: Target Tensor
    sparse_tensor  - numpy.array(Node, points_per_day, days) :: Tensor need to Complete (0 -> missing)
    time_lags  -  numpy.array(len) :: Autoregressive lag set
    alpha  -  non-negative weight parameters for each mode
    rho0  -  learning rate of ADMM algorithm
    lambda0  - λ0 = c0 · ρ with c0 being a constant determining the relative weight of time series regression
    theta  -  truncation factor
    epsilon  -  Epsilon of iteration stops
    maxiter  -  Maximum Iterations
    K  -  unfolding order
    
    Returns:
    Completed Tensor  -  numpy.array(Node, points_per_day, days)
    """

    dim = np.array(sparse_tensor.shape)
    dim_time = np.int32(np.prod(dim) / dim[0])
    d = len(time_lags)
    max_lag = np.max(time_lags)
    sparse_mat = ten2mat(sparse_tensor, 0)
    pos_missing = np.where(sparse_mat == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    dense_test = dense_tensor[pos_test]
    del dense_tensor

    T = np.zeros(dim)
    Z_tensor = sparse_tensor.copy()
    Z = sparse_mat.copy()
    A = 0.001 * np.random.rand(dim[0], d)
    Psis = generate_Psi(dim_time, time_lags)
    iden = sparse.coo_matrix((np.ones(dim_time), (np.arange(0, dim_time), np.arange(0, dim_time))),
                             shape=(dim_time, dim_time))
    it = 0
    ind = np.zeros((d, dim_time - max_lag), dtype=np.int_)
    for i in range(d):
        ind[i, :] = np.arange(max_lag - time_lags[i], dim_time - time_lags[i])
    last_mat = sparse_mat.copy()
    snorm = np.linalg.norm(sparse_mat, 'fro')
    rho = rho0
    while True:
        temp = []
        for m in range(dim[0]):
            Psis0 = Psis.copy()
            for i in range(d):
                Psis0[i + 1] = A[m, i] * Psis[i + 1]
            B = Psis0[0] - sum(Psis0[1:])
            temp.append(B.T @ B)
        for k in range(K):
            rho = min(rho * 1.05, 1e5)
            tensor_hat = np.zeros(dim)
            for p in range(len(dim)):
                tensor_hat += alpha[p] * mat2ten(svt_tnn(ten2mat(Z_tensor - T / rho, p),
                                                         alpha[p] / rho, theta), dim, p)
            temp0 = rho / lambda0 * ten2mat(tensor_hat + T / rho, 0)
            mat = np.zeros((dim[0], dim_time))
            for m in range(dim[0]):
                mat[m, :] = spsolve(temp[m] + rho * iden / lambda0, temp0[m, :])
            Z[pos_missing] = mat[pos_missing]
            Z_tensor = mat2ten(Z, dim, 0)
            T = T + rho * (tensor_hat - Z_tensor)
        for m in range(dim[0]):
            A[m, :] = np.linalg.lstsq(Z[m, ind].T, Z[m, max_lag:], rcond=None)[0]
        mat_hat = ten2mat(tensor_hat, 0)
        tol = np.linalg.norm((mat_hat - last_mat), 'fro') / snorm
        last_mat = mat_hat.copy()
        it += 1
        if it % 200 == 0:
            print_result(it, tol, dense_test, tensor_hat[pos_test])
        if (tol < epsilon) or (it >= maxiter):
            break
    print_result(it, tol, dense_test, tensor_hat[pos_test])

    return tensor_hat


import time
import numpy as np
import argparse
import configparser

np.random.seed(1000)

missing_rate = 0.9

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str, help="configuration file path")
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

points_per_day = int(data_config['points_per_day'])
c = float(training_config['c'])
theta = int(training_config['theta'])
use_nni = int(training_config['use_nni'])

graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
miss_signal_matrix_filename = data_config['miss_graph_signal_matrix_filename']
test_ratio = float(data_config['test_ratio'])

dense_tensor = np.load(graph_signal_matrix_filename, allow_pickle=True)['data'][:, :, 0].astype(np.float32)
T,N = dense_tensor.shape
test_len = int((T//points_per_day)*test_ratio)*points_per_day

dense_tensor = dense_tensor[-test_len:,...].transpose(1, 0).reshape(N, points_per_day, -1)
sparse_tensor = np.load(miss_signal_matrix_filename, allow_pickle=True)['data'][-test_len:, :, 0].transpose(1, 0).reshape(N, points_per_day, -1).astype(np.float32)

dense_tensor = np.nan_to_num(dense_tensor,nan=0)
sparse_tensor = np.nan_to_num(sparse_tensor,nan=0)

if use_nni:
    params = nni.get_next_parameter()
    c = float(params['c'])
    theta = int(params['theta'])

start = time.time()
time_lags = np.arange(1, 7)
alpha = np.ones(3) / 3
rho = 1e-5
lambda0 = c * rho
print(c)
print(theta)
tensor_hat = latc(dense_tensor, sparse_tensor, time_lags, alpha, rho, lambda0, theta)
end = time.time()
print('Running time: %d seconds' % (end - start))




