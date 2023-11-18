import numpy as np
from torch.utils.data import DataLoader, Dataset

class Traffic_Dataset(Dataset):
    def __init__(self,observed_values,values,observed_masks,isTest=False,eval_length=12):
        self.eval_length = eval_length

        self.observed_values = observed_values
        self.observed_masks = observed_masks
        self.gt_masks = observed_masks

        if isTest:  
            self.observed_values = values
            self.observed_masks = np.ones_like(self.gt_masks)

    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.observed_values)

def get_sample_by_overlaped_Sliding_window(X,Y,  mask, sample_len):
    #X,Y,mask: shape(T,N*F)
    X_window,Y_window ,mask_window = [], [], []
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])
        Y_window.append(Y[i:i+sample_len])
        mask_window.append(mask[i:i+sample_len])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)

    return X_window,Y_window, mask_window

def get_dataloader(true_datapath,miss_datapath,val_ratio,test_ratio, batch_size=16,eval_length=12):

    miss = np.load(miss_datapath)
    observed_masks = np.nan_to_num(miss['mask'][:1000, :, 0])
    
    observed_values = np.nan_to_num(miss['data'][:1000, :, 0].astype(np.float32))

    values = np.nan_to_num(np.load(true_datapath)['data'][:1000, :, 0].astype(np.float32))

    mean = np.mean(values[observed_masks==1])
    std = np.std(values[observed_masks==1])
    values = (values - mean)/std
    observed_values = (observed_values - mean)/std
    observed_values[(1-observed_masks).astype(np.bool_)] = 0
    
    T,N = observed_values.shape

    val_len = int(T * val_ratio)
    test_len = int(T * test_ratio)

    train_X, val_X, test_X = observed_values[ :-(val_len+test_len)], \
                             observed_values[ -(val_len+test_len):-(test_len)],\
                             observed_values[-test_len:]
    
    train_Y, val_Y, test_Y = values[ :-(val_len+test_len)], \
                             values[ -(val_len+test_len):-(test_len)],\
                             values[-test_len:]

    train_mask, val_mask, test_mask = observed_masks[ :-(val_len+test_len)], \
                                      observed_masks[ -(val_len+test_len):-(test_len)],\
                                      observed_masks[-test_len:]

    train_X, train_Y,  train_mask = get_sample_by_overlaped_Sliding_window(train_X, train_Y,  train_mask, eval_length)
    val_X, val_Y,  val_mask = get_sample_by_overlaped_Sliding_window( val_X, val_Y,  val_mask, eval_length)
    test_X, test_Y, test_mask = get_sample_by_overlaped_Sliding_window(test_X, test_Y, test_mask, eval_length)

    dataset = Traffic_Dataset(
        train_X,train_Y,train_mask,isTest=False,eval_length=12
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Traffic_Dataset(
        val_X,val_Y,val_mask,isTest=False,eval_length=12
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Traffic_Dataset(
        test_X,test_Y,test_mask,isTest=True,eval_length=12
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader,N,std,mean
