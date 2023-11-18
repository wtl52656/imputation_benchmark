import os
import time

import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, path):
        super(MySet, self).__init__()
        self.content = open(path).readlines()  # 一个用户的数据是一行，也就是说，一个用户的数据就是一个sample
        print(len(self.content))

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        return rec

def collate_fn(recs):  # 处理一个batch的数据
    # print("collate: ", len(recs))
    forward = list(map(lambda x: x['forward'], recs))  # python3.x必须用list将iterators转换为一个list()
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        if "true_data" in recs[0].keys():
            true_data = torch.FloatTensor(list(map(lambda r: r['true_data'], recs)))
            return {'values': values, 'masks': masks, 'deltas': deltas, 'true_data': true_data}
        else:
            return {'values': values, 'masks': masks,  'deltas': deltas}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    return ret_dict

def get_loader(path, batch_size = 16, shuffle = True):
    data_set = MySet(path)

    # print("batch_size: ", batch_size)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 0, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )   # # num_worker=4: DataLoader worker (pid(s) 18819) exited unexpectedly

    return data_iter