# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index

from model import Generator,Discriminator
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import configparser
import argparse
from data_loader import load_data
import nni
import copy

def test(data,test_net):
    test_net.eval()
    with torch.no_grad():
        truth_list = []
        imputed_list = []

        for batch_idx, (X,M_mb) in enumerate(data):

          Z_mb = torch.tensor(uniform_sampler(0, 0.01, X.shape[0], X.shape[1])).cuda()
          X_mb = M_mb * X + (1-M_mb) * Z_mb

          imputed_data = test_net(X_mb,M_mb)

          truth_list.append(X[M_mb==0])
          imputed_list.append(imputed_data[M_mb==0])

        
        truth = renormalization(torch.cat(truth_list).view(-1,1),norm_parameters).view(-1)
        imputed = renormalization(torch.cat(imputed_list).view(-1,1),norm_parameters).view(-1)

        MAE = torch.abs(truth - imputed).mean()
        RMSE = torch.sqrt(((truth - imputed)**2).mean())
        MAPE = torch.divide(torch.abs(truth - imputed),truth).nan_to_num(posinf=0).mean()

        return MAE, RMSE, MAPE
        


def train (train_loader,val_loader,test_loader):
    '''Impute missing values in data_x

    Args:
      - loader: data loader
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        - norm_parameters: norm_parameters
        - learning_rate
        
    Returns:
      - imputed_data: imputed data
    '''

    dim = h_dim

    G = Generator(dim,h_dim).to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

    D = Discriminator(dim,h_dim).to(device)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

    best_mae = .0
    count = 0

    start_train = time.time()
    # Start Iterations
    for it in range(epoch):
        G.train(), D.train()
        G_loss_total = .0
        D_loss_total = .0
        for batch_idx,(X_mb, M_mb) in enumerate(train_loader):

            # Sample random vectors  
            Z_mb = torch.tensor(uniform_sampler(0, 0.01, batch_size, dim)).cuda()
            # Sample hint vectors
            H_mb_temp = torch.tensor(binary_sampler(hint_rate, batch_size, dim)).cuda()
            H_mb = M_mb * H_mb_temp
              
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
            
            G_sample = G(X_mb, M_mb)

            D_prob = D(G_sample.detach(), H_mb)

            D_loss_curr = -(M_mb * torch.log(D_prob + 1e-8) \
                                        + (1-M_mb) * torch.log(1. - D_prob + 1e-8)).mean()

            D_optimizer.zero_grad()
            D_loss_curr.backward()
            D_optimizer.step()                  

            
            G_loss_curr = -((1-M_mb) * torch.log(D_prob.detach() + 1e-8)).mean()
            MSE_loss_curr = ((M_mb * X_mb - M_mb * G_sample)**2).mean() / (M_mb.mean())
            G_loss = G_loss_curr + alpha * MSE_loss_curr

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            G_loss_total += G_loss.item()
            D_loss_total += D_loss_curr.item()
        print(f"================================epoch{it}=======================================")
        print(f"train  Generator_losss: {G_loss_total.__format__('.6f')}\tDiscriminator_loss: {D_loss_total.__format__('.6f')}")

        val_mae,val_rmse,val_mape = test(val_loader,G)

        print(f"val  mae: {val_mae.__format__('.6f')}\trmse: {val_rmse.__format__('.6f')}\tmape: {val_mape.__format__('.6f')}\n")

        if val_mae<best_mae or best_mae == 0 :
            best_mae = val_mae
            best_model_G = copy.deepcopy(G.state_dict())
            best_model_D = copy.deepcopy(D.state_dict())
        else :
            count += 1
        
        if use_nni:
            nni.report_intermediate_result(val_mae.cpu().numpy().item())

        if count == patience :
            break
    
    train_time = time.time() - start_train

    G.load_state_dict(best_model_G)

    start_test_time = time.time()
    test_mae,test_rmse,test_mape = test(test_loader,G)
    test_time = time.time() - start_test_time

    print(f"test  mae: {test_mae.__format__('.6f')}\trmse: {test_rmse.__format__('.6f')}\tmape: {test_mape.__format__('.6f')}\n")
    print(f"test_time: {test_time.__format__('.6f')}\ttrain_time: {train_time.__format__('.6f')}")

    if use_nni:
        nni.report_final_result(test_mae.cpu().numpy().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    ctx = config['train']['cuda']
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    device = torch.device('cuda')

    data_prefix = config['file']['data_prefix']
    save_prefix = config['file']['save_prefix']

    use_nni = int(config['train']['use_nni'])

    savepath = os.path.join(save_prefix, f"{config['train']['type']}_{config['train']['miss_rate']}")
    true_datapath = os.path.join(data_prefix,f"true_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")

    batch_size = int(config['train']['batch_size'])
    hint_rate = float(config['train']['hint_rate'])
    learning_rate = float(config['train']['learning_rate'])
    epoch = int(config['train']['epoch'])
    alpha = float(config['train']['alpha'])

    val_ratio = float(config['train']['val_ratio'])
    test_ratio = float(config['train']['test_ratio'])
    patience = int(config['train']['patience'])
    sample_len = int(config['train']['sample_len'])

    if use_nni:
        params = nni.get_next_parameter()
        alpha = int(params['alpha'])
        config['train']['alpha'] = str(alpha)
        learning_rate = float(params['learning_rate'])
        config['train']['learning_rate'] = str(learning_rate)
        hint_rate = float(params['hint_rate'])
        config['train']['hint_rate'] = str(hint_rate)


    train_loader, val_loader, test_loader, norm_parameters, h_dim = load_data(true_datapath,miss_datapath,val_ratio,\
                                                                      test_ratio,batch_size,sample_len)

    train(train_loader,val_loader,test_loader)
