import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import os

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data,observed_masks, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    test_mask = observed_masks[-T*test_days:]

    val_data = data[-T*(test_days + val_days): -T*test_days]
    val_mask = observed_masks[-T*(test_days + val_days): -T*test_days]

    train_data = data[:-T*(test_days + val_days)]
    train_mask = observed_masks[:-T*(test_days + val_days)]

    return train_data, val_data, test_data

def split_data_by_ratio(data,observed_masks, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    test_mask = observed_masks[-int(data_len*test_ratio):]

    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    val_mask = observed_masks[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]

    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    train_mask = observed_masks[:-int(data_len*(test_ratio+val_ratio))]

    return train_data, val_data, test_data, train_mask, val_mask, test_mask

def data_loader(X, Y,M, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorBool = torch.cuda.BoolTensor if cuda else torch.BoolTensor
    X, Y  = TensorFloat(X), TensorFloat(Y)
    M = TensorBool(M)
    data = torch.utils.data.TensorDataset(X, Y, M)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_randmask( observed_mask):
    rand_for_mask = np.random.randn(*observed_mask.shape) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()  # missing ratio
        num_observed = observed_mask[i].sum().item()
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


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):

    true_datapath = os.path.join(args.data_prefix,f"true_data_{args.type}_{args.miss_rate}_v2.npz")
    miss_datapath = os.path.join(args.data_prefix,f"miss_data_{args.type}_{args.miss_rate}_v2.npz")
    #load raw st dataset
    observed_masks,values = load_st_dataset(true_datapath,miss_datapath)        # T, N, D
    #normalize st data
    values, scaler = normalize_dataset(values, normalizer, args.column_wise)

    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test,mask_train, mask_val, mask_test \
                            = split_data_by_days(values,observed_masks, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test,mask_train, mask_val, mask_test \
                            = split_data_by_ratio(values,observed_masks, args.val_ratio, args.test_ratio)
    
    condmask_train = get_hist_mask(mask_train).astype(int)
    gtmask_train = (mask_train - condmask_train).astype(int)

    condmask_val = get_hist_mask(mask_val).astype(int)
    gtmask_val = (mask_val - condmask_val).astype(int)

    condmask_test = mask_test.astype(int)
    gtmask_test = (1 - condmask_test).astype(int)

    #add time window
    x_tra, y_tra, m_tra = Add_Window_Horizon(data_train,condmask_train,gtmask_train, args.seq_len)
    x_val, y_val, m_val = Add_Window_Horizon(data_val,condmask_val,gtmask_val,args.seq_len)
    x_test, y_test, m_test = Add_Window_Horizon(data_test,condmask_test,gtmask_test, args.seq_len)
    print('Train: ', x_tra.shape, y_tra.shape,m_tra.shape)
    print('Val: ', x_val.shape, y_val.shape,m_val.shape)
    print('Test: ', x_test.shape, y_test.shape,m_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra,m_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val,m_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test,m_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler
