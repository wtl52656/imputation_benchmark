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

import numpy as np

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

def test(loader,test_net):
    with torch.no_grad():
        test_net.eval()
        with torch.no_grad():
            truth_list = []
            imputated_list = []

            for batch_idx, (miss_data,true_data,mask,time_delta) in enumerate(loader):
                noise = torch.normal(mean=0,std=0.01,size=(miss_data.shape[0],miss_data.shape[1],miss_data.shape[2])).cuda()
                imputated_data =test_net(miss_data,mask, noise, time_delta)

                truth_list.append(true_data[mask==0])
                imputated_list.append(imputated_data[mask==0])

            truth = torch.cat(truth_list).view(-1)*std +mean
            imputated = torch.cat(imputated_list).view(-1)*std +mean

            MAE = torch.abs(truth - imputated).mean()
            RMSE = torch.sqrt(((truth - imputated)**2).mean())
            MAPE = torch.divide(torch.abs(truth - imputated),truth).nan_to_num(posinf=0).mean()

            return MAE, RMSE, MAPE
        
def pretrain(model,opt,loader):
    l2loss = nn.MSELoss()
    model.train()
    for it in range(pretrain_epoch):
        epoch_loss = 0
        for batch_idx,(miss_data,true_data,mask,time_delta) in enumerate(loader):
            noise = torch.normal(mean=0,std=0.01,size=(miss_data.shape[0],miss_data.shape[1],miss_data.shape[2])).cuda()
            imputated_data = model(miss_data,mask, noise, time_delta)

            opt.zero_grad()
            loss = l2loss(imputated_data[mask.bool()],miss_data[mask.bool()])
            loss.backward()
            opt.step()
            epoch_loss+=loss.item()
        print(f"================================epoch{it}=======================================")
        print(f"pretrain  Generator_losss: {epoch_loss.__format__('.6f')}\n")
        


def train (train_loader,val_loader,test_loader):

    G = Generator(input_size,h_dim,z_dim, sample_len).to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

    D = Discriminator(input_size,h_dim).to(device)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

    start_train = time.time()

    pretrain(G,G_optimizer,train_loader)

    best_mae = .0
    count = 0
    l2loss = nn.MSELoss()
    # Start Iterations
    for it in range(epoch):
        G.train(), D.train()
        G_loss_total = .0
        D_loss_total = .0
        for batch_idx,(miss_data,true_data,mask,time_delta) in enumerate(train_loader):

            noise = torch.normal(mean=0,std=0.01,size=(miss_data.shape[0],miss_data.shape[1],miss_data.shape[2])).cuda()
            imputated_data = G(miss_data,mask, noise, time_delta)

            imputated_time_delta = torch.ones_like(time_delta).cuda()

            D_prob_real = D(miss_data,time_delta).mean()
            D_prob_fake = D(imputated_data,imputated_time_delta).mean()

            G_loss = -D_prob_fake + alpha*l2loss(imputated_data[mask.bool()],miss_data[mask.bool()])

            D_loss = -D_prob_real + D_prob_fake

            if (it+1)%disc_iters == 0:
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()                  
            else :
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
            
            G_loss_total += G_loss.item()
            D_loss_total += D_loss.item()
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
            nni.report_intermediate_result(val_mae.item())

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
        nni.report_final_result(best_mae.item())


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
    learning_rate = float(config['train']['learning_rate'])
    epoch = int(config['train']['epoch'])
    pretrain_epoch = int(config['train']['pretrain_epoch'])
    disc_iters = int(config['train']['disc_iters'])
    alpha = float(config['train']['alpha'])
    z_dim = int(config['train']['z_dim'])
    h_dim = int(config['train']['h_dim'])

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
        z_dim = int(params['z_dim'])
        config['train']['z_dim'] = str(z_dim)
        h_dim = int(params['h_dim'])
        config['train']['h_dim'] = str(h_dim)


    train_loader, val_loader, test_loader, mean, std, input_size = load_data(true_datapath,miss_datapath,val_ratio,\
                                                                      test_ratio,batch_size,sample_len)
    train(train_loader,val_loader,test_loader)
