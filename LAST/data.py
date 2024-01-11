import numpy as np
import os
import pandas as pd
import argparse
import configparser
import warnings
import datetime
import re

warnings.filterwarnings('ignore')

def SlideWindows(data,condmask,obmask, window=12):

    length = len(data)
    end_index = length  - window + 1

    Datas = [] 
    Cond_masks = []
    Ob_masks = []

    index = 0

    while index < end_index:
        Datas.append(data[index:index+window])
        Cond_masks.append(condmask[index:index+window])
        Ob_masks.append(obmask[index:index+window])
        index = index + 1

    Datas = np.asarray(Datas)
    Cond_masks = np.asarray(Cond_masks)
    Ob_masks = np.asarray(Ob_masks)

    return Datas, Cond_masks, Ob_masks

def get_randmask( observed_mask):
    rand_for_mask = np.random.randn(*observed_mask.shape) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()  # missing ratio
        num_observed = observed_mask[i].sum()
        num_masked = round(num_observed * sample_ratio)
        #rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        rand_for_mask[i][rand_for_mask[i].argsort()[-num_masked:]] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape)
    return cond_mask

def get_hist_mask( observed_mask):
    for_pattern_mask = observed_mask
    rand_mask = get_randmask(observed_mask)

    cond_mask = observed_mask.copy()
    for i in range(len(cond_mask)):
        mask_choice = np.random.rand()
        if mask_choice > 0.5:
            cond_mask[i] = rand_mask[i]
        else:  # draw another sample for histmask (i-1 corresponds to another sample)
            cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
    return cond_mask

def loadImputationData(config):
    data_config = config['Data']

    train_ratio = float(data_config['train_ratio'])
    val_ratio = float(data_config['val_ratio'])
    test_ratio = float(data_config['test_ratio'])
    sample_len = int(data_config['sample_len'])

    miss_type = data_config['miss_type']
    miss_rate = float(data_config['miss_rate'])
    data_prefix = data_config['data_prefix']

    true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")

    miss = np.load(miss_datapath,allow_pickle=True)
    mask = miss['mask'][:, :, :1] 

    true_data = np.load(true_datapath,allow_pickle=True)['data'][:, :, :1].astype(np.float32)
    true_data[np.isnan(true_data)] = 0
    mask*= (true_data!=0)


    # Divide the dataset first ,and construct the sample
    slices = true_data.shape[0]
    train_slices = int(slices * train_ratio)
    val_slices = int(slices * val_ratio)
    test_slices = slices - train_slices - val_slices

    train_set = true_data[ : train_slices]
    train_mask = mask[ : train_slices]
    train_condamask = get_hist_mask(train_mask)
    train_set[train_mask==0] = 0
    print(train_set.shape)

    val_set = true_data[train_slices : val_slices + train_slices]
    val_mask = mask[train_slices : val_slices + train_slices]
    val_condamask = get_hist_mask(val_mask)
    val_set[val_mask==0] = 0
    print(val_set.shape)

    test_set = true_data[-test_slices : ]
    test_mask = (test_set!=0).astype('int32')
    test_condamask = mask[-test_slices : ]
    print(test_set.shape)
    

    x_trains,  cond_trains, ob_trains = SlideWindows(train_set,train_condamask,train_mask,sample_len)
    x_vals, cond_vals, ob_vals = SlideWindows(val_set,val_condamask,val_mask,sample_len)
    x_tests, cond_tests, ob_tests = SlideWindows(test_set,test_condamask,test_mask,sample_len)

    print("train: ", x_trains.shape)
    print("val: ", x_vals.shape)
    print("test: ", x_tests.shape)


    return x_trains,  cond_trains, ob_trains,\
            x_vals,  cond_vals, ob_vals,\
            x_tests,  cond_tests, ob_tests,\
            test_set, test_condamask, test_mask
