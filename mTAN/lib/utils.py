import os
import numpy as np
import torch
import torch.utils.data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)

    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mask_MAPE(y_true, y_pred, mask):
    mask = mask.astype('long')
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs(y_pred - y_true) / y_true
        mask = mask & ~np.isnan(mape) & ~np.isinf(mape)
        mape = np.nan_to_num(mape, posinf=0)
        mape = mask * mape
        return np.sum(mape) / np.sum(mask)


def mask_MAE(y_true, y_pred, mask):
    mae = np.abs(y_true - y_pred)
    mae = mae * mask
    return np.sum(mae) / np.sum(mask)


def mask_RMSE(y_true, y_pred, mask):
    rmse = (y_true - y_pred) ** 2
    rmse = rmse * mask
    return np.sqrt(np.sum(rmse) / np.sum(mask))


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def compute_losses(observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, std, norm, device):
    # pred_x: (k_iwae, B, N, SEQ_LEN,F)
    noise_std_ = torch.zeros(pred_x.size()).to(device) + std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1) #(k_iwae, B, N),损失函数的第一项，observed_data在该分布下的期望概率
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)#与标准正太分布的kl散度
    logpx = logpx.reshape(logpx.shape[0], -1)
    analytic_kl = analytic_kl.reshape(-1)
    if norm:
        c = observed_mask.sum(-1).sum(-1).reshape(-1)
        analytic_kl = analytic_kl[c!=0]
        logpx = logpx[:, c!=0]
        c = c[c!=0]
        logpx /= c
        analytic_kl /= c
    return logpx, analytic_kl


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_sample_by_overlaped_Sliding_window(X, Y,  mask,num_ref_points, sample_len):
    #X,Y,mask: shape(T,N,1)
    X_window,Y_window, mask_window = [], [], []
    timestamp_window, query_point_window = [], []
    for i in range(X.shape[0]-sample_len+1):
        X_window.append(X[i:i+sample_len])
        Y_window.append(Y[i:i+sample_len])
        mask_window.append(mask[i:i+sample_len])

        timestamp=np.linspace(0.0,1.0,sample_len)   #(seq_len) 样本的时间戳

        query_point = np.linspace(
            0, 1, num_ref_points)  #(num_ref_points) 样本的reference point

        timestamp_window.append(np.repeat(np.expand_dims(
                                timestamp, axis=-1), X.shape[1], axis=-1))  
        query_point_window.append(np.repeat(np.expand_dims(
                                query_point, axis=-1), X.shape[1], axis=-1)) 

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)
    timestamp_window = np.array(timestamp_window)#(B,N,seq_len)
    query_point_window = np.array(query_point_window)#(B,N,num_ref_points)

    return X_window,Y_window, mask_window, timestamp_window, query_point_window


def load_data (true_datapath,miss_datapath,val_ratio,test_ratio,num_ref_points,sample_len=12):
    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, :1]  #(T,N,F)
    miss_data = miss['data'][:, :, :1]

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :1]
    true_data[np.isnan(true_data)] = 0

    mask = np.expand_dims(mask.reshape((mask.shape[0],-1)),1) 
    true_data = np.expand_dims(true_data.reshape((true_data.shape[0],-1)),1) 
    miss_data = np.expand_dims(miss_data.reshape((miss_data.shape[0],-1)),1) 

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

    train_X, train_Y,  train_mask, train_timestamp, train_query_point = \
        get_sample_by_overlaped_Sliding_window(train_X, train_Y,  train_mask,num_ref_points, sample_len)


    val_X, val_Y, val_mask, val_timestamp, val_query_point = \
        get_sample_by_overlaped_Sliding_window(val_X, val_Y,  val_mask,num_ref_points, sample_len)


    test_X, test_Y,  test_mask, test_timestamp, test_query_point = \
        get_sample_by_overlaped_Sliding_window(test_X, test_Y,  test_mask,num_ref_points, sample_len)


    file_data = {}

    file_data['max'] = np.max(np.reshape(true_data,(-1,true_data.shape[-1])),axis=0)
    file_data['min'] = np.min(np.reshape(true_data,(-1,true_data.shape[-1])),axis=0)

    #把数据格式改成(B,N,SEQ_LEN,F)
    file_data['train_x'], file_data['train_target'], file_data['train_mask'], file_data['train_timestamp'], file_data['train_query'] = \
        train_X.transpose(0,2,1,3), train_Y.transpose(0,2,1,3), train_mask.transpose(0,2,1,3), train_timestamp.transpose(0,2,1), train_query_point.transpose(0,2,1)

    file_data['val_x'], file_data['val_target'], file_data['val_mask'], file_data['val_timestamp'], file_data['val_query'] = \
        val_X.transpose(0,2,1,3), val_Y.transpose(0,2,1,3), val_mask.transpose(0,2,1,3), val_timestamp.transpose(0,2,1), val_query_point.transpose(0,2,1)

    file_data['test_x'], file_data['test_target'], file_data['test_mask'], file_data['test_timestamp'], file_data['test_query'] = \
        test_X.transpose(0,2,1,3), test_Y.transpose(0,2,1,3), test_mask.transpose(0,2,1,3), test_timestamp.transpose(0,2,1), test_query_point.transpose(0,2,1)
    

    return file_data



def load_graphdata_normY_channel1(true_datapath,miss_datapath,sample_len,val_ratio,test_ratio,num_ref_points, batch_size, shuffle=True):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file_data = load_data (true_datapath,miss_datapath,val_ratio,test_ratio,num_ref_points,sample_len)

    input_dim = file_data['train_x'].shape[-1]

    train_x = file_data['train_x']  # (B,N,SEQ_LEN,F)
    train_target = file_data['train_target']  # (B,N,SEQ_LEN,F)
    train_mask = file_data['train_mask']  # (B,N,SEQ_LEN,F)
    train_timestamp = file_data['train_timestamp']  # (B,SEQ_LEN)
    train_query = file_data['train_query'] # (B,num_ref_points)

    val_x = file_data['val_x']
    val_target = file_data['val_target']
    val_mask = file_data['val_mask']
    val_timestamp = file_data['val_timestamp']
    val_query = file_data['val_query']

    test_x = file_data['test_x']
    test_target = file_data['test_target']
    test_mask = file_data['test_mask']
    test_timestamp = file_data['test_timestamp']
    test_query = file_data['test_query']

    _max = file_data['max']  # (F)
    _min = file_data['min']  # (F)

    # 统一进行归一化，变成[-1,1]之间的值
    train_x = max_min_normalization(train_x, _max, _min)
    test_x = max_min_normalization(test_x, _max, _min)
    val_x = max_min_normalization(val_x, _max, _min)
    train_target_norm = max_min_normalization(train_target, _max, _min)
    test_target_norm = max_min_normalization(test_target, _max, _min)
    val_target_norm = max_min_normalization(val_target, _max, _min)

    train_x_tensor = torch.from_numpy(train_x).type(
        torch.FloatTensor)  # (B, N, SEQ_LEN, F)
    train_target_tensor = torch.from_numpy(train_target_norm).type(
        torch.FloatTensor)  # (B, N, SEQ_LEN, F)
    train_mask_tensor = torch.from_numpy(
        train_mask).type(torch.IntTensor)
    train_timestamp_tensor = torch.from_numpy(
        train_timestamp).type(torch.FloatTensor)  
    train_query_tensor = torch.from_numpy(
        train_query).type(torch.FloatTensor)  

    train_dataset = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor, train_mask_tensor, train_timestamp_tensor, train_query_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_x_tensor = torch.from_numpy(val_x).type(
        torch.FloatTensor)  # (B, N, SEQ_LEN, F)

    val_target_tensor = torch.from_numpy(val_target_norm).type(
        torch.FloatTensor)  # (B, N, SEQ_LEN, F)
    val_mask_tensor = torch.from_numpy(
        val_mask).type(torch.IntTensor)
    val_timestamp_tensor = torch.from_numpy(
        val_timestamp).type(torch.FloatTensor)
    val_query_tensor = torch.from_numpy(
        val_query).type(torch.FloatTensor)  

    val_dataset = torch.utils.data.TensorDataset(
        val_x_tensor, val_target_tensor, val_mask_tensor, val_timestamp_tensor, val_query_tensor)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size)

    test_x_tensor = torch.from_numpy(test_x).type(
        torch.FloatTensor)    # (B, N, SEQ_LEN, F)
    test_target_tensor = torch.from_numpy(test_target_norm).type(
        torch.FloatTensor)    # (B, N, SEQ_LEN, F)
    test_mask_tensor = torch.from_numpy(
        test_mask).type(torch.IntTensor)  
    test_timestamp_tensor = torch.from_numpy(
        test_timestamp).type(torch.FloatTensor) 
    test_query_tensor = torch.from_numpy(
        test_query).type(torch.FloatTensor)  

    test_dataset = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor, test_mask_tensor, test_timestamp_tensor, test_query_tensor)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)

    print('train:', train_x_tensor.size(), train_target_tensor.size(),
          train_mask_tensor.size(), train_timestamp_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size(),
          val_mask_tensor.size(), val_timestamp_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size(),
          test_mask_tensor.size(), test_timestamp_tensor.size())

    return train_loader, val_loader, test_loader, _max, _min,input_dim

# load_graphdata_normY_channel1("../data/PEMS04/true_data.npz", 1, 16, shuffle=True)