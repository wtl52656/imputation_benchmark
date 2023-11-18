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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
import torch

from utils import normalization

def get_sample_by_overlaped_Sliding_window(X,  mask, sample_len):
    #X,Y,mask: shape(T,N,1)
    X_window, mask_window = [], []
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])
        mask_window.append(mask[i:i+sample_len])

    X_window = np.array(X_window)
    mask_window = np.array(mask_window)

    return X_window, mask_window


def data_loader(X,  mask, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X,  mask = TensorFloat(X),  TensorFloat(mask)
    data = torch.utils.data.TensorDataset(X,  mask)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def load_data (true_datapath,miss_datapath,val_ratio,test_ratio,batch_size,sample_len=12):
    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, 0] 

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, 0]
    true_data[np.isnan(true_data)] = 0

    true_data , norm_parameters = normalization(true_data)

    dim = true_data.shape[1]*sample_len

    val_len = int(true_data.shape[0] * val_ratio)
    test_len = int(true_data.shape[0] * test_ratio)

    train_X, val_X, test_X = true_data[ :-(val_len+test_len)], \
                             true_data[ -(val_len+test_len):-(test_len)],\
                             true_data[-test_len:]

    train_mask, val_mask, test_mask = mask[ :-(val_len+test_len)], \
                             mask[ -(val_len+test_len):-(test_len)],\
                             mask[-test_len:]
    
    print(train_X.shape,val_X.shape,test_X.shape)

    train_X,  train_mask = get_sample_by_overlaped_Sliding_window(train_X,  train_mask, sample_len)
    train_X, train_mask = train_X.reshape(train_X.shape[0],-1), train_mask.reshape(train_mask.shape[0],-1)
    train_loader = data_loader(train_X,  train_mask,batch_size)

    val_X, val_mask = get_sample_by_overlaped_Sliding_window(val_X,  val_mask, sample_len)
    val_X, val_mask = val_X.reshape(val_X.shape[0],-1), val_mask.reshape(val_mask.shape[0],-1)
    val_loader = data_loader(val_X, val_mask,batch_size)

    test_X,  test_mask = get_sample_by_overlaped_Sliding_window(test_X,  test_mask, sample_len)
    test_X,  test_mask = test_X.reshape(test_X.shape[0],-1), test_mask.reshape(test_mask.shape[0],-1)
    test_loader = data_loader(test_X,  test_mask,batch_size)

    return train_loader, val_loader, test_loader, norm_parameters, dim