import argparse
import configparser
import numpy as np
import torch
import torch.optim as optim
from time import time
import os
import copy
import nni
import models
from lib.utils import load_graphdata_normY_channel1, compute_losses, re_max_min_normalization, mask_MAE, mask_RMSE, mask_MAPE


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./configurations/PEMS04.conf', type=str, help="configuration file path")
args = parser.parse_args()


config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
model_config = config['Model']
training_config = config['Training']

input_dim = int(data_config['input-dim'])

device = torch.device('cuda:0')
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
dropout = float(training_config['dropout'])
seed = int(training_config['seed'])
norm = bool(training_config['norm'])

std = float(model_config['std'])
latent_dim = int(model_config['latent-dim'])
rec_hidden = int(model_config['rec-hidden'])
gen_hidden = int(model_config['gen-hidden'])
embed_time = int(model_config['embed-time'])
k_iwae = int(model_config['k-iwae'])                   #插补时从分布中采样次数
num_ref_points = int(model_config['num-ref-points'])   #注意力机制中的reference point，减少计算量
enc_num_heads = int(model_config['enc-num-heads'])
dec_num_heads = int(model_config['dec-num-heads'])
kl = bool(model_config['kl'])#损失函数包括得出的分布与正太分布之间的kl散度

data_prefix = data_config['data_prefix']
save_prefix = data_config['save_prefix']
savepath = os.path.join(save_prefix, f"{data_config['type']}_{data_config['miss_rate']}")
true_datapath = os.path.join(data_prefix,f"true_data_{data_config['type']}_{data_config['miss_rate']}_v2.npz")
miss_datapath = os.path.join(data_prefix,f"miss_data_{data_config['type']}_{data_config['miss_rate']}_v2.npz")


val_ratio = float(training_config['val_ratio'])
test_ratio = float(training_config['test_ratio'])
patience = int(training_config['patience'])
sample_len = int(data_config['sample_len'])
use_nni = int(training_config['use_nni'])

if use_nni:
    params = nni.get_next_parameter()

    rec_hidden = int(params['rec_hidden'])
    model_config['rec-hidden'] = str(rec_hidden)

    gen_hidden = int(params['gen-hidden'])
    model_config['gen-hidden'] = str(gen_hidden)

    enc_num_heads = int(params['enc-num-heads'])
    model_config['enc-num-heads'] = str(enc_num_heads)

    dec_num_heads = int(params['dec-num-heads'])
    model_config['dec-num-heads'] = str(dec_num_heads)


def compute_val_loss(model, val_loader):
    model.eval() # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        loss, n = 0, 0
        start_time = time()
        for batch_index, batch_data in enumerate(val_loader):
            x, target, mask, timestamp, query = [d.to(device) for d in batch_data]
            batch_len = x.shape[0]
            pred_x, qz0_mean, qz0_logvar = model(x, mask, timestamp, query)
            logpx, analytic_kl = compute_losses(
                x, mask, qz0_mean, qz0_logvar, pred_x, std, norm, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean() - np.log(k_iwae))
            loss += loss.item() * batch_len
            n += batch_len
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' %
                      (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' % (time()-start_time))

        val_loss = loss / n
    return val_loss


def compute_test_metrics(model, test_loader):
    model.eval() # ensure dropout layers are in evaluation mode
    imputation, truth, loss_mask = [], [], []
    with torch.no_grad():
        for batch_data in test_loader:
            x, target, mask, timestamp, query = [d.to(device) for d in batch_data]
            pred_x, _, _ = model(x, mask, timestamp, query)
            pred_x = re_max_min_normalization(pred_x.mean(0).cpu().data.numpy(), _max, _min)#多个分布取均值后反归一化
            target = re_max_min_normalization(target.cpu().data.numpy(), _max, _min)
            imputation.append(pred_x)
            truth.append(target)
            loss_mask.append((1-mask).cpu().data.numpy())
    imputation = np.concatenate(imputation, axis=0)
    truth = np.concatenate(truth, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    rmse = mask_RMSE(truth, imputation, loss_mask)
    mae = mask_MAE(truth, imputation, loss_mask)
    mape = mask_MAPE(truth, imputation, loss_mask)
    return rmse, mae, mape


if __name__ == '__main__':
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    '''
    train_loader, val_loader, test_loader, _max, _min,input_dim = \
    load_graphdata_normY_channel1(true_datapath,miss_datapath,sample_len,val_ratio,test_ratio,num_ref_points, batch_size)

    model = models.enc_dec_mtan(input_dim, latent_dim, rec_hidden, gen_hidden, embed_time, enc_num_heads, dec_num_heads, k_iwae, True, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_start_time = time()

    best_val_loss = np.inf
    for epoch in range(0, epochs):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl = 0, 0
        if kl:
            wait_until_kl_inc = 10
            if epoch < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (epoch - wait_until_kl_inc))
        else:
            kl_coef = 1
        model.train()
        for train_batch in train_loader:
            x, target, mask, timestamp, query = [d.to(device) for d in train_batch]
            batch_len = x.shape[0]

            pred_x, qz0_mean, qz0_logvar = model(x, mask, timestamp, query)
            #pre_x从分布中采样出来的多个插补值；qz0_mean, qz0_logvar：隐变量分布

            logpx, analytic_kl = compute_losses(
                x, mask, qz0_mean, qz0_logvar, pred_x, std, norm, device)
            #logpx用于计算实际值和插补值分布的误差，analytic_kl表示求出的分布与正态分布的kl散度
            
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean() - np.log(k_iwae))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len #插补期望均值
            avg_kl += torch.mean(analytic_kl) * batch_len#kl散度均值
        val_loss = compute_val_loss(model, val_loader)
        if use_nni:
            nni.report_intermediate_result(val_loss.item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print('Iter: {},Train avg elbo: {:.6f},\t avg reconst: {:.6f},\t avg kl: {:.6f}'
            .format(epoch, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n))
        print('Iter: {},Val loss: {:.6f}'
            .format(epoch, val_loss))
    
    print('train cost time: %.4fs' % (time()-train_start_time))

    if use_nni:
        nni.report_final_result(best_val_loss.item())

    model.load_state_dict( best_model_state_dict)
    best_model = model
    print('Val Best epoch: {}, Best loss: {:.6f}'.format(best_epoch, best_val_loss))
    rmse, mae, mape = compute_test_metrics(best_model, test_loader)
    print('TEST\tMAE: {:.6f},\t RMSE: {:.6f},\t MAPE: {:.6f}'.format( mae,rmse, mape))
    

