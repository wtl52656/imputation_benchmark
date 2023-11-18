import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
import os
from tqdm import tqdm
from utils import get_randmask, get_block_mask

def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


class Survey_Dataset(Dataset):
    def __init__(self,true_data,ob_mask,gt_mask, c_data,val_start, test_start, eval_length=12, mode="train", missing_pattern='block',
                 is_interpolate=False, target_strategy='random', missing_ratio=None):
        self.eval_length = eval_length
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.missing_pattern = missing_pattern

        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = ob_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
        elif mode == 'test':
            self.observed_mask = np.ones_like(ob_mask[test_start:])
            self.gt_mask = ob_mask[test_start:]
            self.observed_data = true_data[test_start:]
            
        self.current_length = len(self.observed_mask) - eval_length + 1
        self.use_index = np.arange(self.current_length)
        self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]
        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy,min_seq=3, max_seq=12)
            else:
                cond_mask = get_randmask(ob_mask_t)
        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[index],
            "cond_mask": cond_mask.numpy()
        }
        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
        return s

    def __len__(self):
        return self.current_length


def get_dataloader(batch_size, device, val_len=0.2, test_len=0.2, missing_pattern='block',
                   is_interpolate=False, num_workers=4, target_strategy='random',data_prefix='',miss_type='SR-TR',miss_rate=0.1):
    
    true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")


    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, 0] 

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, 0]
    true_data[np.isnan(true_data)] = 0
    train_mean, train_std= np.mean(true_data[mask==1]), np.std(true_data[mask==1])
    true_data = (true_data - train_mean)/train_std

    ob_mask = mask.astype('uint8')
    c_data = np.copy(true_data) * ob_mask

    T,N = true_data.shape
    print(f'raw data shape:{ true_data.shape}')

    
    
    if missing_pattern == 'block':
        eval_mask = sample_mask(shape=(T, N), p=0.0015, p_noise=0.05, min_seq=3, max_seq=12)
    elif missing_pattern == 'point':
        eval_mask = sample_mask(shape=(T, N), p=0., p_noise=0.25, min_seq=3, max_seq=12)

    gt_mask = (1-(eval_mask | (1-ob_mask))).astype('uint8')

    val_start = int((1 - val_len - test_len) * T)
    test_start = int((1 - test_len) * T)



    dataset = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="train", 
                              missing_pattern=missing_pattern,
                             is_interpolate=is_interpolate, target_strategy=target_strategy)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    print(f'train dataset len:{dataset.__len__()}')


    dataset_test = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="test", 
                                   missing_pattern=missing_pattern,
                                  is_interpolate=is_interpolate, target_strategy=target_strategy)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print(f'test dataset len:{dataset_test.__len__()}')

    dataset_valid = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="valid", 
                                   missing_pattern=missing_pattern,
                                   is_interpolate=is_interpolate, target_strategy=target_strategy)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print(f'val dataset len:{dataset_valid.__len__()}')

    scaler = torch.tensor(train_std).to(device).float()
    mean_scaler = torch.tensor(train_mean).to(device).float()

    print(f'scaler: {scaler}')
    print(f'mean_scaler: {mean_scaler}')

    return train_loader, valid_loader, test_loader, scaler, mean_scaler,N


def get_test_randmask(observed_mask, missing_ratio):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = missing_ratio  # missing ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

