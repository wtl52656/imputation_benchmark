#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.GCASTN import make_model
from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel1,MaskL1Loss,MaskL2Loss
from tensorboardX import SummaryWriter
from prepareData import read_and_generate_dataset_encoder_decoder
from tqdm import tqdm



best_epoch = 0
# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']

miss_type = data_config['miss_type']
miss_rate = float(data_config['miss_rate'])
data_prefix = graph_signal_matrix_filename

true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")
all_datapath = os.path.join(data_prefix,f"gcastn_data_{miss_type}_{miss_rate}_v2.npz")

if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch']) 
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

if os.path.exists(all_datapath):
    print(f'read data from {all_datapath}')
    all_data = np.load(all_datapath)
else:
    all_data = read_and_generate_dataset_encoder_decoder(all_datapath,true_datapath,miss_datapath, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)


# direction = 1 means: if i connected to j, adj[i,j]=1;
# direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
if direction == 2:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
if direction == 1:
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)
folder_dir = miss_type+ ' ' + str(miss_rate) + ' ' + folder_dir
if aware_temporal_context:
    folder_dir = folder_dir+'Tcontext'
if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('./experiments', dataset_name, folder_dir)

# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor,train_mask_tensor, val_loader, val_target_tensor,val_mask_tensor, test_loader, test_target_tensor,test_mask_tensor, _max, _min = load_graphdata_normY_channel1(
    all_data, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout, aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

print(net, flush=True)


def train_main():
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",start_epoch)
    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    #criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    #criterion = MaskL1Loss().to(DEVICE)
    criterion = MaskL2Loss().to(DEVICE)
    mseloss = nn.MSELoss().to(DEVICE)
    lamta = 1
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    total_param = 0

    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        #print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    #for var_name in optimizer.state_dict():
        #print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set


        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch,DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)


        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()


        for batch_index, batch_data in tqdm(enumerate(train_loader)):

            encoder_inputs1, encoder_inputs2,decoder_inputs1, decoder_inputs2, labels,mask,timestemps,delta1,delta2 = batch_data


            encoder_inputs1 = encoder_inputs1.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)
            encoder_inputs2 = encoder_inputs2.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs1 = decoder_inputs1.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)
            decoder_inputs2 = decoder_inputs2.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)
            mask = mask.unsqueeze(-1).to(DEVICE)
            timestemps = timestemps.unsqueeze(-1).to(DEVICE)
            delta1 = delta1.unsqueeze(-1).to(DEVICE)
            delta2 = delta2.unsqueeze(-1).to(DEVICE)
            optimizer.zero_grad()
            outputs1 = net(encoder_inputs1,mask,decoder_inputs1,timestemps,delta1)
            outputs2 = net(encoder_inputs2, mask, decoder_inputs2, timestemps, delta2)

            loss1,_ = criterion(outputs1, labels,mask)
            loss2, _ = criterion(outputs2, labels, mask)
            loss3 = mseloss(outputs1,outputs2)
            loss = loss1 + loss2 + lamta * loss3

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

    #print('best epoch:', best_epoch, flush=True)


    #print('apply the best val model on the test data set ...', flush=True)

    #predict_main(best_epoch, test_loader, test_target_tensor,test_mask_tensor, _max, _min, 'test')

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs1, encoder_inputs2, decoder_inputs1, decoder_inputs2, labels, mask, timestemps, delta1, delta2 = batch_data

            encoder_inputs1 = encoder_inputs1.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)
            encoder_inputs2 = encoder_inputs2.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs1 = decoder_inputs1.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)
            decoder_inputs2 = decoder_inputs2.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)
            mask = mask.unsqueeze(-1).to(DEVICE)
            timestemps = timestemps.unsqueeze(-1).to(DEVICE)
            delta1 = delta1.unsqueeze(-1).to(DEVICE)
            delta2 = delta2.unsqueeze(-1).to(DEVICE)


            predict_length = labels.shape[2]  # T

            optimizer.zero_grad()

            encoder_inputs = torch.cat([encoder_inputs1,encoder_inputs2],dim=0)
            mask = torch.cat([mask,mask],dim=0)
            timestemps = torch.cat([timestemps,timestemps],dim=0)
            delta = torch.cat([delta1,delta2],dim=0)
            decoder_inputs = torch.cat([decoder_inputs1,decoder_inputs2],dim=0)
            labels = torch.cat([labels,labels],dim=0)

            encoder_output = net.encode(encoder_inputs,mask,timestemps,delta)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss,_ = criterion(predict_output, labels,mask)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch,DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, test_target_tensor,test_mask_tensor, _max, _min, 'test')
    #predict_main(best_epoch, val_loader, val_target_tensor,val_mask_tensor, _max, _min, 'test')


def predict_main(epoch, data_loader, data_target_tensor,data_mask_tensor, _max, _min, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor,data_mask_tensor, epoch, _max, _min, params_path, type,DEVICE)


if __name__ == "__main__":

    train_main()
    #predict_main(52, test_loader, test_target_tensor,test_mask_tensor, _max, _min, 'test')















