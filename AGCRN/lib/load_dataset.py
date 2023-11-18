import os
import numpy as np

def load_st_dataset(true_datapath,miss_datapath):
    

    miss = np.load(miss_datapath)
    observed_masks = np.nan_to_num(miss['mask'][:, :, :1])
    
    values = np.nan_to_num(np.load(true_datapath)['data'][:, :, :1].astype(np.float32))

    return observed_masks,values
