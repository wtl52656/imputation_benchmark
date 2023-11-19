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

def get_sample_by_overlaped_Sliding_window(X, Y,  mask, sample_len):
    #X,Y,mask: shape(T,N,1) -> (B,sample_len,N,1)
    X_window,Y_window, mask_window = [], [], []
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])
        Y_window.append(Y[i:i+sample_len])
        mask_window.append(mask[i:i+sample_len])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)

    return X_window,Y_window, mask_window


def data_loader(X, Y,  mask, time_delta, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y,  mask, time_delta = TensorFloat(X), TensorFloat(Y), TensorFloat(mask), TensorFloat(time_delta)
    data = torch.utils.data.TensorDataset(X, Y,  mask, time_delta)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def load_data (true_datapath,miss_datapath,val_ratio,test_ratio,batch_size,sample_len=12):
    """
    Load data / generate sample and time delta
    time delta : Time delta between observable points
    
    Parameters: 
    true_datapath  - path of ground-truth
    miss_datapath  - path of the data with missing

    Returns:

    Sample shape:[Batch,sample_len,N*F]
    input_size: N*F
    """
    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, :1] 
    miss_data = miss['data'][:, :, :1]

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :1]
    true_data[np.isnan(true_data)] = 0

    input_size = true_data.shape[-1]*true_data.shape[-2]

    mean , std = true_data[miss_data.astype(bool)].mean(), true_data[miss_data.astype(bool)].std()

    true_data = (true_data - mean)/std
    miss_data = (miss_data - mean)/std

    val_len = int(true_data.shape[0] * val_ratio)
    test_len = int(true_data.shape[0] * test_ratio)

    train_X, val_X, test_X = miss_data[ :-(val_len+test_len)], \
                             miss_data[ -(val_len+test_len):-(test_len)],\
                             miss_data[-test_len:]

    train_Y, val_Y, test_Y = true_data[ :-(val_len+test_len)], \
                             true_data[ -(val_len+test_len):-(test_len)],\
                             true_data[-test_len:]

    train_mask, val_mask, test_mask = mask[ :-(val_len+test_len)], \
                             mask[ -(val_len+test_len):-(test_len)],\
                             mask[-test_len:]
    
    print(train_X.shape,val_X.shape,test_X.shape)

    train_X, train_Y,  train_mask = get_sample_by_overlaped_Sliding_window(train_X, train_Y,  train_mask, sample_len)
    train_X, train_Y,  train_mask = train_X.reshape(train_X.shape[0],sample_len,-1), train_Y.reshape(train_Y.shape[0],sample_len,-1),  train_mask.reshape(train_mask.shape[0],sample_len,-1)
    train_delta = np.zeros_like(train_mask)
    for i in range(1,sample_len):
        train_delta[:,i] = train_delta[:,i]*(1-train_mask[:,i-1])+1
    train_loader = data_loader(train_X, train_Y,train_mask,train_delta,batch_size)

    val_X, val_Y, val_mask = get_sample_by_overlaped_Sliding_window(val_X, val_Y,  val_mask, sample_len)
    val_X, val_Y, val_mask = val_X.reshape(val_X.shape[0],sample_len,-1), val_Y.reshape(val_Y.shape[0],sample_len,-1),  val_mask.reshape(val_mask.shape[0],sample_len,-1)
    val_delta = np.zeros_like(val_mask)
    for i in range(1,sample_len):
        val_delta[:,i] = val_delta[:,i]*(1-val_mask[:,i-1])+1
    val_loader = data_loader(val_X, val_Y, val_mask,val_delta,batch_size)


    test_X, test_Y,  test_mask = get_sample_by_overlaped_Sliding_window(test_X, test_Y,  test_mask, sample_len)
    test_X, test_Y, test_mask = test_X.reshape(test_X.shape[0],sample_len,-1), test_Y.reshape(test_Y.shape[0],sample_len,-1),  test_mask.reshape(test_mask.shape[0],sample_len,-1)
    test_delta = np.zeros_like(test_mask)
    for i in range(1,sample_len):
        test_delta[:,i] = test_delta[:,i]*(1-test_mask[:,i-1])+1
    test_loader = data_loader(test_X, test_Y,  test_mask,test_delta,batch_size)

    return train_loader, val_loader, test_loader, mean, std, input_size