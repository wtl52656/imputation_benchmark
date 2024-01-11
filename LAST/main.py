import pandas as pd
import numpy as np
import os
import argparse
import configparser
from copy import deepcopy
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from data import loadImputationData

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                    help="configuration file path")
parser.add_argument("--miss_type", default='SR-TR', type=str,
                    help="configuration file path")
parser.add_argument("--miss_rate", default=0.1, type=float)
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(args.config, 'r') as f:
    print(f.read())
config.read(args.config)

data_config = config['Data']
dataset = data_config['dataset_name']
miss_type = args.miss_type  #data_config['miss_type']
miss_rate = args.miss_rate  #float(data_config['miss_rate'])
data_config['miss_type'] = miss_type
data_config['miss_rate'] = str(miss_rate)

print(f"miss_rate = {miss_rate}\nmiss_type = {miss_type}\n")


def MAPE_np(pred, true, mask=None):
    if mask is not None:
        mask = mask.astype('bool')
        true = true[mask]
        pred = pred[mask]

    return np.mean(np.absolute(np.divide((true - pred), true)))


def Last(input,cond_mask,ob_mask):
    start = time.time()
    x = input.copy()
    x[cond_mask == 0] = 0

    eval_mask = ob_mask - cond_mask

    B,T,N,F = x.shape

    
    y_hat = x.copy()

    last_data = np.zeros_like(x[:,0,...])
    for i in range(T): #往前找第一个未缺失的点
        y_hat[:,i,...] = np.where(y_hat[:,i,...]==0,last_data,y_hat[:,i,...])
        last_data = np.where(x[:,i,...]!=0,x[:,i,...],last_data)

    last_data = np.zeros_like(x[:,0,...])
    for i in range(T-1,-1,-1):#往后找第一个未缺失的点
        y_hat[:,i,...] = np.where(y_hat[:,i,...]==0,last_data,y_hat[:,i,...])
        last_data = np.where(x[:,i,...]!=0,x[:,i,...],last_data)
    
    MAE, RMSE, MAPE = mean_absolute_error(input.reshape(-1,1),y_hat.reshape(-1,1),sample_weight=eval_mask.reshape(-1,1)),\
                      mean_squared_error(input.reshape(-1,1),y_hat.reshape(-1,1),sample_weight=eval_mask.reshape(-1,1))**0.5,\
                      MAPE_np(y_hat.reshape(-1,1),input.reshape(-1,1),eval_mask.reshape(-1,1))

    return y_hat,MAE, RMSE, MAPE,time.time()-start

def Run(config,miss_type,miss_rate):
    results = []
    x_trains,  cond_trains, ob_trains,\
            x_vals,  cond_vals, ob_vals,\
            x_tests,  cond_tests, ob_tests ,\
            test_set, test_condamask, test_mask    = loadImputationData(config)

    B,T,N,F = x_tests.shape

    y_hat,MAE, RMSE, MAPE, time_cost = Last(x_tests,cond_tests,ob_tests)

    print(f"\nDataset: {dataset}  misstype: {miss_type}  missrate: {miss_rate}  (Completion by sample separately):")
    print(f"MAE: {MAE}")
    print(f"RMSE: {RMSE}")
    print(f"MAPE: {MAPE}")
    print(f"time: {round(time_cost,2)} seconds\n")
    results.append([round(MAE,2), round(RMSE,2), round(MAPE*100,2), round(time_cost,2)])


    y_hat,MAE, RMSE, MAPE, time_cost = Last(test_set.reshape(1,-1,N,F),test_condamask.reshape(1,-1,N,F),test_mask.reshape(1,-1,N,F))

    print(f"\nDataset: {dataset}  misstype: {miss_type}  missrate: {miss_rate}  (Completing the test set together):")
    print(f"MAE: {MAE}")
    print(f"RMSE: {RMSE}")
    print(f"MAPE: {MAPE}")
    print(f"time: {round(time_cost,2)} seconds\n")
    results.append([round(MAE,2), round(RMSE,2), round(MAPE*100,2), round(time_cost,2)])

    return results

if __name__ == '__main__':
    Run(config,miss_type,miss_rate)



