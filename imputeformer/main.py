import torch
import argparse
import configparser
from data import load_data,inverse_Sliding_window
import os
import time
from pypots.imputation import ImputeFormer
from pypots.utils.metrics import calc_mae,calc_rmse
import numpy as np
import nni

def masked_mape_np(y_pred, y_true,  indicating_mask, null_val=np.nan):
    y_pred = np.where(indicating_mask,y_pred,null_val)
    y_true = np.where(indicating_mask,y_true,null_val)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask & np.greater(y_true,1e-4)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", default='/mnt/nfsData17/ZhaoMiaomiao1/miss_data/', type=str, help="data file path")
parser.add_argument("--dataset", default='PEMS04', type=str, help="dataset")
parser.add_argument("--logs", default='./logs', type=str, help="log path")
parser.add_argument("--miss_type", default='SR-TR', type=str, help="miss_type")
parser.add_argument("--miss_rate", default=0.5, type=float, help="miss_rate")
parser.add_argument("--val_ratio", default=0.2, type=float, help="val_ratio")
parser.add_argument("--test_ratio", default=0.2, type=float, help="test_ratio")
parser.add_argument("--sample_len", default=12, type=int, help="sample_len")
parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
parser.add_argument("--use_nni", default=0, type=int, help="use_nni")
args = parser.parse_args()


if args.use_nni:
    params = nni.get_next_parameter()
    args.dataset = params['dataset']
    args.miss_type = params['miss_type']
    args.miss_rate = params['miss_rate']

log_root = os.path.join(args.logs,f"{args.dataset}_{args.miss_type}_{args.miss_rate}_{time.ctime()}")
os.mkdir(log_root)

import yaml
model_config_path =  os.path.join("./configurations",f"{args.dataset}.yaml")
with open(model_config_path) as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
# model_config['epochs'] = 2
# model_config['patience'] = 1

true_datapath = os.path.join(args.data_prefix,f"{args.dataset}/true_data_{args.miss_type}_{args.miss_rate}_v2.npz")
miss_datapath = os.path.join(args.data_prefix,f"{args.dataset}/miss_data_{args.miss_type}_{args.miss_rate}_v2.npz")

train_set,val_set,test_set,feature_dim,mean , std = load_data(true_datapath,miss_datapath,args.val_ratio,args.test_ratio,args.sample_len)

start = time.time()
model = ImputeFormer(n_steps=args.sample_len,n_features=feature_dim,**model_config,batch_size=args.batch_size,saving_path=log_root,device='cuda',ORT_weight=0)
model.fit(train_set,val_set)
train_time = time.time() - start


start = time.time()
imputation = model.impute(test_set)*std + mean
test_time = time.time() - start

test_ori = test_set['X_ori']*std + mean
indicating_mask = np.isnan(test_ori) ^ np.isnan(test_set['X'])


mae = calc_mae(imputation, np.nan_to_num(test_ori), indicating_mask) 
rmse = calc_rmse(imputation, np.nan_to_num(test_ori), indicating_mask)
mape = masked_mape_np(imputation,test_ori,indicating_mask)

print(log_root)
print(f'{"mae":<12}{"rmse":<12}{"mape":<12}')
print(f'{mae:<12.4f}{rmse:<12.4f}{mape:<12.4f}')
print("GPU memory: ",torch.cuda.max_memory_allocated()/(1024*1024))
print(f"train time: {train_time}\n" + f"test time: {test_time}\n")

imputation = inverse_Sliding_window(imputation)
ground_truth = inverse_Sliding_window(test_ori)

result_npz = os.path.join(log_root,f"result.npz")
np.savez_compressed(result_npz, imputation=imputation, ground_truth=ground_truth)

result_txt = f'{"mae":<12}{"rmse":<12}{"mape":<12}\n'+f'{mae:<12.4f}{rmse:<12.4f}{mape:<12.4f}\n' \
            + f"GPU memory: {torch.cuda.max_memory_allocated()/(1024*1024)}\n"  \
            + f"train time: {train_time}\n" + f"test time: {test_time}\n"

with open(os.path.join(log_root,f"logs.log"), 'w') as f:
    print(result_txt, file=f)

if args.use_nni:
    nni.report_final_result(mae)