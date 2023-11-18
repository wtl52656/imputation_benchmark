import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from metrics import masked_mae, masked_mape, masked_rmse, masked_mse
from utils import *
import random
from basic_structure import IGNNK
import argparse
import sys
import os
from tqdm import tqdm
import copy
import configparser
import nni
import datetime
import time

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvconfig

parser = argparse.ArgumentParser()

parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

ctx = config['train']['cuda']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx

data_prefix = config['file']['data_prefix']
save_prefix = config['file']['save_prefix']
distance_df_filename = config['file']['distance_df_filename']

use_nni = int(config['train']['use_nni'])

num_of_vertices = int(config['train']['num_of_vertices'])

#number of seeing node
no = int(config['train']['no'])

#number of masked node
nm = int(config['train']['nm'])


#sampled time dimension
time_dim = int(config['train']['time_dim'])

#hidden dimension for graph convolution in time
hidden_dim = int(config['train']['hidden_dim'])

#If using diffusion convolution, the actual diffusion convolution step is K+1
K = int(config['train']['K'])

#max training episode
max_iter = int(config['train']['max_iter'])

#the learning_rate for Adam optimizer
learning_rate = float(config['train']['learning_rate'])

#the max value from experience
E_maxvalue = float(config['train']['E_maxvalue'])

batch_size = int(config['train']['batch_size'])

test_ratio = float(config['train']['test_ratio'])
val_ratio = float(config['train']['val_ratio'])

patience = int(config['train']['patience']) #early stop patience

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^config


if use_nni:
    params = nni.get_next_parameter()
    hidden_dim = int(params['hidden_dim'])
    config['train']['hidden_dim'] = str(hidden_dim)
    learning_rate = float(params['learning_rate'])
    config['train']['learning_rate'] = str(learning_rate)
    K = int(params['K'])
    config['train']['K'] = str(K)
    no = int(params['no_nm'][0])
    config['train']['no'] = str(no)
    nm = int(params['no_nm'][1])
    config['train']['nm'] = str(nm)

no_nm = no + nm


def get_sample_by_overlaped_Sliding_window(X, Y, mask, sample_len=12):
    #X,Y,mask: shape(N,T,1)
    X_window, Y_window, mask_window = [], [], []
    for i in range(X.shape[1]-sample_len+1):
        X_window.append(X[:,i:i+sample_len])
        Y_window.append(Y[:,i:i+sample_len])
        mask_window.append(mask[:,i:i+sample_len])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)

    return X_window, Y_window, mask_window

def data_loader(X, Y, mask, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y, mask = TensorFloat(X), TensorFloat(Y), TensorFloat(mask)
    data = torch.utils.data.TensorDataset(X, Y, mask)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def load_data(true_datapath,miss_datapath, distance_df_filename):
    # X, Y, mask: N x T x 1
    X, Y, mask, A_q, A_h = load_pems_data(true_datapath,miss_datapath, distance_df_filename,num_of_vertices)

    val_len = int(X.shape[1] * val_ratio)
    test_len = int(X.shape[1] * test_ratio)

    train_X, val_X, test_X = X[:, :-(val_len+test_len)], X[:, -(val_len+test_len):-(test_len)], X[:,-test_len:]
    train_Y, val_Y, test_Y = Y[:, :-(val_len+test_len)], Y[:, -(val_len+test_len):-(test_len)], Y[:,-test_len:]
    train_mask, val_mask, test_mask = mask[:, :-(val_len+test_len)], mask[:, -(val_len+test_len):-(test_len)], mask[:,-test_len:]

    train_X, train_Y, train_mask = get_sample_by_overlaped_Sliding_window(train_X, train_Y, train_mask, time_dim)
    train_loader = data_loader(train_X, train_Y, train_mask,batch_size)

    val_X, val_Y, val_mask = get_sample_by_overlaped_Sliding_window(val_X, val_Y, val_mask, time_dim)
    val_loader = data_loader(val_X, val_Y, val_mask,batch_size)

    test_X, test_Y, test_mask = get_sample_by_overlaped_Sliding_window(test_X, test_Y, test_mask, time_dim)
    test_loader = data_loader(test_X, test_Y, test_mask,batch_size)

    return train_loader, val_loader, test_loader, A_q, A_h


"""
Define the test error
"""


def test_error(STmodel, test_loader, A_q, A_h):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    STmodel.eval()
    with torch.no_grad():
        Mf_A_q = torch.from_numpy(A_q).to(device)
        Mf_A_h = torch.from_numpy(A_h).to(device)

        truth = []
        o = []
        masks = []
        for batch_idx, (data, target,mask) in enumerate(test_loader):
            imputation = STmodel(data/E_maxvalue, Mf_A_q, Mf_A_h)
            o.append(imputation)
            truth.append(target)
            masks.append(mask)
        
        o = torch.cat(o,dim = 0).cpu().numpy()
        o = o*E_maxvalue

        truth = torch.cat(truth,dim = 0).cpu().numpy()
        masks = torch.cat(masks,dim = 0).cpu().numpy()
        loss_mask = 1 - masks

        MAE = masked_mae(truth, o, loss_mask)
        RMSE = masked_rmse(truth, o, loss_mask)
        MAPE = masked_mape(truth, o, loss_mask)

    STmodel.train()
    return MAE, RMSE, MAPE

if __name__ == "__main__":
    """
    Model training
    """
    savepath = os.path.join(save_prefix, f"{config['train']['type']}_{config['train']['miss_rate']}")
    true_datapath = os.path.join(data_prefix,f"true_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")

    train_loader, val_loader, test_loader, A_q, A_h = load_data(
        true_datapath,miss_datapath, distance_df_filename)
    # The graph neural networks
    device = torch.device('cuda')
    STmodel = IGNNK(time_dim, hidden_dim, K).to(device)
    criterion = MaskL2Loss() 
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    MAE_list = []
    best_mae = np.inf

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    count = 0
    train_start = time.time()
    train_epoch_time = 0
    for epoch in range(max_iter):
        train_loss = 0
        start = time.time()
        for batch_idx, (data, target,mask) in enumerate(train_loader):

            know_mask = set(random.sample(range(0, data.shape[1]), no_nm))
            
            inputs = data[:,list(know_mask)]/E_maxvalue
            labels = target[:,list(know_mask)]/E_maxvalue
            masks =  mask[:,list(know_mask)]

            if (masks.sum() == 0):
                continue

            missing_index = torch.rand(size=(inputs.shape), dtype=torch.float32)
            missing_index = (missing_index > (nm/no_nm)).int().cuda()
            zeros = torch.zeros_like(inputs,dtype=torch.float32)

            Mf_inputs = torch.where(missing_index==0,zeros, inputs)

            Mf_A_q = torch.from_numpy(
                A_q[list(know_mask), :][:, list(know_mask)]).to(device)
            Mf_A_h = torch.from_numpy(
                A_h[list(know_mask), :][:, list(know_mask)]).to(device)
            
            optimizer.zero_grad()
            # Obtain the reconstruction
            X_res = STmodel(Mf_inputs, Mf_A_q, Mf_A_h)
            
            loss = criterion(labels, X_res,masks)
            
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_epoch_time += time.time() - start

        MAE,RMSE,MAPE = test_error(STmodel, val_loader, A_q, A_h)
        if use_nni:
            nni.report_intermediate_result(MAE)
        
        print(f"epoch{epoch} train loss:{train_loss}")
        print(f"epoch{epoch} val \t mae:{MAE} \t RMSE:{RMSE} \t MAPE:{MAPE} \n")

        MAE_list.append(MAE)
        if MAE < best_mae:
            best_mae = MAE
            best_epoch = epoch
            best_model = copy.deepcopy(STmodel)
            count = 0
        else :
            count += 1
        
        if count > patience:
            print("early stop")
            break

    train_end = time.time()
    print("tarin+val time: {:.2f} seconds,\t tarin epoch average time: {:.2f} seconds:".format(train_end - train_start,train_epoch_time/(epoch+1)))

    test_start = time.time()

    best_mae, best_rmse, best_mape = test_error(
        best_model, test_loader, A_q, A_h)

    test_end = time.time()

    print("best epoch: {:d}, best_MAE: {:.2f}, best_RMSE: {:.2f}, best_MAPE: {:.2f}".format(
        best_epoch, best_mae, best_rmse, best_mape*100))
    print("test time: {:.2f} seconds".format(test_end - test_start))
    
    if use_nni:
        nni.report_final_result(best_mae)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    torch.save(STmodel.state_dict(), os.path.join(
                savepath, f"best_model_{current_time}"))
