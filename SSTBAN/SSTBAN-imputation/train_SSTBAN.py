import pandas as pd
import numpy as np
import os
import argparse
import configparser
import warnings
import torch
from copy import deepcopy
import time
import torch.utils.data
import torch.optim as optim
from model.sstban_model import SSTBAN, make_model
import time
import datetime
import math
import torch.nn as nn
import nni
from lib import sstban_utils
from prepare.prepareData_SSTBAN import loadImputationData
import pickle
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(args.config, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)

data_config = config['Data']
training_config = config['Training']

miss_type = data_config['miss_type']
miss_rate = float(data_config['miss_rate'])

# Data config
if config.has_option('Data', 'graph_signal_matrix_filename'):
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
else:
    graph_signal_matrix_filename = None

dataset_name = data_config['dataset_name']
print("dataset_name: ", dataset_name)
num_of_vertices = int(data_config['num_of_vertices'])
time_slice_size = int(data_config['time_slice_size'])

# nni
use_nni = int(training_config['use_nni'])
mode = training_config['mode']
ctx = training_config['ctx']
if use_nni:
    import nni
    params = nni.get_next_parameter()
    L = int(params['L'])
    training_config['L'] = str(L)
    K = int(params['K'])
    training_config['K'] = str(K)
    d = int(params['d'])
    training_config['d'] = str(d)
    miss_rate = float(params['node_miss_rate'])
    training_config['node_miss_rate']=str(miss_rate)
    T_miss_len=int(params['T_miss_len'])
    training_config['T_miss_len']=str(T_miss_len)
    self_weight_dis = float(params['self_weight_dis'])
    training_config['self_weight_dis'] = str(self_weight_dis)
else:
    L = int(training_config['L'])
    K = int(training_config['K'])
    d = int(training_config['d'])
    self_weight_dis = float(training_config['self_weight_dis'])
    reference = int(training_config['reference'])

# Training config
learning_rate = float(training_config['learning_rate'])
max_epoch = int(training_config['epochs'])
decay_epoch = int(training_config['decay_epoch'])
batch_size = int(training_config['batch_size'])
num_his = int(data_config['sample_len'])
num_pred = int(data_config['sample_len'])
patience = int(training_config['patience'])
in_channels = int(training_config['in_channels'])

# load dataset

x_trains, te_trains, cond_trains, ob_trains,\
            x_vals, te_vals, cond_vals, ob_vals,\
            x_tests, te_tests, cond_tests, ob_tests,\
            data_mean, data_std = loadImputationData(config)

x_trains = torch.from_numpy(x_trains.astype('float32')).type(torch.FloatTensor)
x_vals = torch.from_numpy(x_vals.astype('float32')).type(torch.FloatTensor)
x_tests = torch.from_numpy(x_tests.astype('float32')).type(torch.FloatTensor)

te_trains = torch.from_numpy(te_trains.astype('int32'))
te_vals = torch.from_numpy(te_vals.astype('int32'))
te_tests = torch.from_numpy(te_tests.astype('int32'))

cond_trains = torch.from_numpy(cond_trains.astype('float32')).type(torch.FloatTensor)
ob_trains = torch.from_numpy(ob_trains.astype('float32')).type(torch.FloatTensor)

cond_vals = torch.from_numpy(cond_vals.astype('float32')).type(torch.FloatTensor)
ob_vals = torch.from_numpy(ob_vals.astype('float32')).type(torch.FloatTensor)

cond_tests = torch.from_numpy(cond_tests.astype('float32')).type(torch.FloatTensor)
ob_tests = torch.from_numpy(ob_tests.astype('float32')).type(torch.FloatTensor)


trainX, trainTE, trainCond, trainOb  = x_trains, te_trains, cond_trains, ob_trains
valX, valTE, valCond, valOb  = x_vals, te_vals, cond_vals, ob_vals
testX, testTE, testCond, testOb  = x_tests, te_tests, cond_tests, ob_tests

#select device
gpu = int(training_config['gpu'])
if gpu:
    USE_CUDA=torch.cuda.is_available()
    if USE_CUDA:
        print("CUDA:", USE_CUDA, ctx)
        torch.cuda.set_device(int(ctx))
        device=torch.device("cuda")
    else:
        print("NO CUDA,Let's use cpu!")
        device = torch.device("cpu")
else:
    device=torch.device("cpu")
    print("Use CPU")
model = make_model(config, bn_decay=0.1)
model = model.to(device)
mean_ = torch.tensor(data_mean).to(device)
std_ = torch.tensor(data_std).to(device)
parameters = sstban_utils.count_parameters(model)
print('trainable parameters: {:,}'.format(parameters))

# train
print('Start training ...')

val_loss_min = float('inf')
wait = 0
best_model_wts = None
best_model = deepcopy(model.state_dict())
best_epoch = -1
best_loss = np.inf

num_train = x_trains.shape[0]
num_val = x_vals.shape[0]
num_test = x_tests.shape[0]
train_num_batch = math.ceil(num_train / batch_size)
val_num_batch = math.ceil(num_val / batch_size)
test_num_batch = math.ceil(num_test / batch_size)
loss_criterion = nn.L1Loss()
loss_criterion_self = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=decay_epoch,
                                      gamma=0.9)

if use_nni:
    exp_id = nni.get_experiment_id()
    trail_id = nni.get_trial_id()
    dataset_name = dataset_name + '_' + str(exp_id) + '_' + str(trail_id)
exp_datadir="experiments/SSTBAN/"
if not os.path.exists(exp_datadir):
    os.makedirs(exp_datadir)
params_filename = os.path.join(exp_datadir, f"{dataset_name}_{K}_{L}_{d}_{miss_type}_{miss_rate}_best_params")
train_time_epochs = []
val_time_epochs=[]
total_start_time = time.time()
for epoch_num in range(0, max_epoch):
    if wait >= patience:
        print(f'early stop at epoch: {epoch_num}, the val loss is {val_loss_min}')
        break
    # shuffle
    permutation = torch.randperm(num_train)

    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainCond = trainCond[permutation]
    trainOb = trainOb[permutation]

    start_train = time.time()
    model.train()
    train_loss = 0

    print(f"epoch {epoch_num} start!")
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx].to(device)
        TE = trainTE[start_idx: end_idx].to(device)
        label = trainX[start_idx: end_idx].detach().clone().to(device)
        Cond_mask = trainCond[start_idx: end_idx].to(device)
        Ob_mask = trainOb[start_idx: end_idx].to(device)
        eval_point = (Ob_mask-Cond_mask).to(device)

        X = (X-mean_)/std_

        optimizer.zero_grad()
        pred,complete_X_enc,X_miss = model(X, TE,mode,Cond_mask,Ob_mask)
        # self_label=X[...,0]
        # self_label=self_label*std_[0]+mean_[0]
        # self_pred=self_pred*std_[0]+mean_[0]
        pred = pred * std_ + mean_
        # print("pred:", pred.shape, "label: ", label.shape)
        loss_self=loss_criterion_self(complete_X_enc*eval_point,X_miss*eval_point)
        loss_batch = loss_criterion(pred*eval_point, label*eval_point)
        train_loss += float(loss_batch) * (end_idx - start_idx)
        loss_all=(1-self_weight_dis)*loss_batch+self_weight_dis*loss_self
        loss_all.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print(f'Training batch: {batch_idx + 1} in epoch:{epoch_num}, training batch loss:{loss_batch:.4f}')
        del X, TE, label, pred, loss_batch
    train_loss /= num_train
    end_train = time.time()

    print("evaluating on valid set now!")
    val_loss = 0
    start_val = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)

            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            label = valX[start_idx: end_idx].detach().clone().to(device)
            Cond_mask = valCond[start_idx: end_idx].to(device)
            Ob_mask = valOb[start_idx: end_idx].to(device)
            eval_point = (Ob_mask-Cond_mask).to(device)

            X = (X-mean_)/std_
            pred,self_pred= model(X, TE,'test',Cond_mask)
            pred = pred * std_ + mean_
            
            loss_batch = loss_criterion(pred*eval_point, label*eval_point)
            val_loss += loss_batch * (end_idx - start_idx)
            del X, TE, label, pred, loss_batch
    val_loss /= num_val
    end_val = time.time()

    if use_nni:
        nni.report_intermediate_result(val_loss.item())

    train_time_epochs.append(end_train - start_train)
    val_time_epochs.append(end_val - start_val)
    print('%s | epoch: %04d/%d, training time: %.1fs, validation time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch_num + 1,
         max_epoch, end_train - start_train, end_val - start_val))
    print(f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
    if val_loss <= val_loss_min:
        wait = 0
        val_loss_min = val_loss
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch_num
    else:
        wait += 1
    scheduler.step()
params_filename=params_filename+"_"+str(val_loss_min.cpu().numpy())
torch.save(best_model, params_filename)
print(f"saving model to {params_filename}")


print("train one epoch for average: ", np.array(train_time_epochs).mean())
print("valid one epoch for average: ", np.array(val_time_epochs).mean())
print("train time: ", time.time() - total_start_time, "s")

# evaluation
print('evaluating on all set now!')
#print('paramfile:',params_filename)
model.load_state_dict(best_model)
model.eval()


with torch.no_grad():

    trainPred = []
    trainLabel = []
    trainEval = []
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)

        X = trainX[start_idx: end_idx].to(device)
        TE = trainTE[start_idx: end_idx].to(device)
        label = trainX[start_idx: end_idx].detach().clone().to(device)
        Cond_mask = trainCond[start_idx: end_idx].to(device)
        Ob_mask = trainOb[start_idx: end_idx].to(device)
        eval_point = (Ob_mask-Cond_mask)

        X = (X-mean_)/std_
        pred_batch,self_pred = model(X, TE,'test',Cond_mask)

        trainPred.append((pred_batch* std_+ mean_).detach().cpu().numpy())
        trainLabel.append(label.detach().cpu().numpy())
        trainEval.append(eval_point.detach().cpu().numpy())

        del X, TE, pred_batch,label,Cond_mask,Ob_mask
    trainPred = torch.from_numpy(np.concatenate(trainPred, axis=0))
    trainLabel = torch.from_numpy(np.concatenate(trainLabel, axis=0))
    trainEval = torch.from_numpy(np.concatenate(trainEval, axis=0)).type(torch.BoolTensor)

    valPred = []
    valLabel = []
    valEval = []
    for batch_idx in range(val_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_val, (batch_idx + 1) * batch_size)

        X = valX[start_idx: end_idx].to(device)
        TE = valTE[start_idx: end_idx].to(device)
        label = valX[start_idx: end_idx].detach().clone().to(device)
        Cond_mask = valCond[start_idx: end_idx].to(device)
        Ob_mask = valOb[start_idx: end_idx].to(device)
        eval_point = (Ob_mask-Cond_mask)

        X = (X-mean_)/std_
        pred_batch,self_pred = model(X, TE,'test',Cond_mask)

        valPred.append((pred_batch* std_ + mean_).detach().cpu().numpy())
        valLabel.append(label.detach().cpu().numpy())
        valEval.append(eval_point.detach().cpu().numpy())

        del X, TE, pred_batch,label,Cond_mask,Ob_mask

    valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
    valLabel = torch.from_numpy(np.concatenate(valLabel, axis=0))
    valEval = torch.from_numpy(np.concatenate(valEval, axis=0)).type(torch.BoolTensor)

    testPred = []
    testLabel = []
    testEval = []
    start_test = time.time()
    for batch_idx in range(test_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_test, (batch_idx + 1) * batch_size)


        X = testX[start_idx: end_idx].to(device)
        TE = testTE[start_idx: end_idx].to(device)
        label = testX[start_idx: end_idx].detach().clone().to(device)
        Cond_mask = testCond[start_idx: end_idx].to(device)
        Ob_mask = testOb[start_idx: end_idx].to(device)
        eval_point = (Ob_mask-Cond_mask)

        X = (X-mean_)/std_
        pred_batch,self_pred = model(X, TE,'test',Cond_mask)

        testPred.append((pred_batch* std_ + mean_).detach().cpu().numpy())
        testLabel.append(label.detach().cpu().numpy())
        testEval.append(eval_point.detach().cpu().numpy())

        del X, TE, pred_batch,label,Cond_mask,Ob_mask

    testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
    testLabel = torch.from_numpy(np.concatenate(testLabel, axis=0))
    testEval = torch.from_numpy(np.concatenate(testEval, axis=0)).type(torch.BoolTensor)

end_test = time.time()
train_mae, train_rmse, train_mape = sstban_utils.mae_rmse_mape(trainPred, trainLabel,trainEval)
val_mae, val_rmse, val_mape = sstban_utils.mae_rmse_mape(valPred, valLabel,valEval)
test_mae, test_rmse, test_mape = sstban_utils.mae_rmse_mape(testPred, testLabel,testEval)


if use_nni:
    nni.report_final_result(test_mae)
print('testing time: %.1fs' % (end_test - start_test))


print('             LOSS\tMAE\tRMSE\tMAPE')
print('train      %.2f\t%.2f\t%.2f\t%.2f%%' %
        (train_mae, train_mae, train_rmse, train_mape))
print('val        %.2f\t%.2f\t%.2f\t%.2f%%' %
        (val_mae, val_mae, val_rmse, val_mape))
print('test       %.2f\t%.2f\t%.2f\t%.2f%%' %
        (test_mae , test_mae, test_rmse, test_mape))
print('performance in each prediction step')


columns = ['loss', 'mae', 'rmse', 'mape']
index = ['train', 'test', 'val']

values = [[train_mae, train_mae, train_rmse, train_mape],
        [val_mae, val_mae, val_rmse, val_mape],
        [test_mae, test_mae, test_rmse, test_mape]]
for i in range(len(values)):
    for j in range(len(values[0])):
        values[i][j] = round(values[i][j], 4)

MAE, RMSE, MAPE = [], [], []
values = []
for step in range(num_pred):
    mae, rmse, mape = sstban_utils.mae_rmse_mape(testPred[:, step], testLabel[:, step],testEval[:,step])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    values.append([mae, rmse, mape])
    print('step: %02d         %.2f\t%.2f\t%.2f%%' %
                   (step + 1, mae, rmse, mape))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
print('average:         %.2f\t%.2f\t%.2f%%' %
             (average_mae, average_rmse, average_mape))


foldername = os.path.join(exp_datadir, f"{dataset_name}_{K}_{L}_{d}_{miss_type}_{miss_rate}_output")
with open(
    foldername  + ".pk", "wb"
) as f:

    pickle.dump(
        [
            testPred,
            testLabel,
            testEval,
            mean_,
            std_
        ],
        f
    )
