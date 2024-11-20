import numpy as np
import torch

def get_sample_by_overlaped_Sliding_window(X, sample_len):
    #X,Y,mask: shape(T,N,1)
    X_window, mask_window = [], []
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])

    X_window = np.array(X_window)

    return X_window

def inverse_Sliding_window(X_window):
    #X:[B,L,F]
    #return [T,F]
    B,L,F = X_window.shape

    X = X_window[0,:,:]

    if B>1:
        rest = X_window[1:,-1,:]
        X = np.concatenate((X,rest),axis=0)

    return X


def load_data(true_datapath,miss_datapath,val_ratio,test_ratio,sample_len=12):
    miss = np.load(miss_datapath,allow_pickle=True)
    mask = miss['mask'][:, :, 0] 


    true_data = np.load(true_datapath,allow_pickle=True)['data'][:, :, 0].astype(np.float32)

    mean , std = true_data[mask.astype(bool)].mean(), true_data[mask.astype(bool)].std()
    true_data = (true_data - mean)/std
    true_data[np.isnan(true_data)] = 0
    
    feature_dim = true_data.shape[-1]

    val_len = int(true_data.shape[0] * val_ratio)
    test_len = int(true_data.shape[0] * test_ratio)

    train_X, val_X, test_X = true_data[ :-(val_len+test_len)], \
                             true_data[ -(val_len+test_len):-(test_len)],\
                             true_data[-test_len:]

    train_mask, val_mask, test_mask = mask[ :-(val_len+test_len)], \
                             mask[ -(val_len+test_len):-(test_len)],\
                             mask[-test_len:]
    
    print(train_X.shape,val_X.shape,test_X.shape)

    train_X = np.where(train_mask,train_X,np.nan)
    train_X= get_sample_by_overlaped_Sliding_window(train_X, sample_len)
    train_set = {"X": train_X}

    val_X_ori = get_sample_by_overlaped_Sliding_window(val_X, sample_len)
    val_X = np.where(val_mask,val_X,np.nan)
    val_X= get_sample_by_overlaped_Sliding_window(val_X, sample_len)
    val_set = {"X": val_X,"X_ori":val_X_ori}

    test_X_ori = get_sample_by_overlaped_Sliding_window(test_X, sample_len)
    test_X = np.where(test_mask,test_X,np.nan)
    test_X= get_sample_by_overlaped_Sliding_window(test_X, sample_len)
    test_set = {"X": test_X,"X_ori":test_X_ori}


    return train_set,val_set,test_set,feature_dim,mean , std