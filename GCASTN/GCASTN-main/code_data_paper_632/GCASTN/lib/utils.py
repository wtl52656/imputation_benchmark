import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


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


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)


        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  #

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # 首先确保A是一个对称矩阵

    D = np.diag(np.sum(A, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - A  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - W  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 求解拉普拉斯矩阵的最大奇异值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj

class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
            loss = (torch.abs(pred - gt) * mask).sum() / mask_sum
            return loss, dict(l1_loss=loss)



class MaskL2Loss(nn.Module):
    def __init__(self):
        super(MaskL2Loss, self).__init__()

    def forward(self, pred, target, mask, detach=True, reduce_batch=True):
        assert mask.max() <= 1 + 1e-6
        if detach:
            target = target.detach()
        mask = mask.detach()
        assert pred.shape == target.shape
        dif = (pred - target) ** 2 * (mask.float())
        loss = torch.sum(dif.reshape(mask.shape[0], -1).contiguous(), 1)
        count = torch.sum(mask.reshape(mask.shape[0], -1).contiguous(), 1).detach()
        loss[count == 0] = loss[count == 0] * 0
        #loss = loss / (count + 1)
        if reduce_batch:
            non_zero_count = torch.sum((count > 0).float())
            if non_zero_count == 0:
                loss = torch.sum(loss) * 0
            else:
                loss = torch.sum(loss) / non_zero_count
            return loss,dict(l2_loss=loss)
        else:
            return loss,dict(l2_loss=loss)
def mask_MAE(y_true,y_pred,mask):
    """

    :param y_true:numpy (n_sample,feature_num)
    :param y_pred:numpy (n_sample,feature_num)
    :param mask:numpy (n_sample,feature_num)
    :return: int
    """
    masked_true = []
    masked_pred = []
    for i in range(len(mask)):
        if (mask[i]==0):
            masked_true.extend([y_true[i]])
            masked_pred.extend([y_pred[i]])
        else:
            continue
    mae = mean_absolute_error(masked_true,masked_pred)

    return mae


def mask_RMSE(y_true, y_pred, mask):
    """
    :param y_true: numpy (n_sample,feature_num)
    :param y_pred: numpy (n_sample,feature_num)
    :param mask: numpy (n_sample,feature_num)
    :return: int
    """

    masked_true = []
    masked_pred = []
    for i in range(len(mask)):
        if (mask[i]==0):
            masked_true.extend([y_true[i]])
            masked_pred.extend([y_pred[i]])
        else:
            continue
    rmse = mean_squared_error(masked_true, masked_pred) ** 0.5

    return rmse


def mask_MAPE(y_true, y_pred, mask):
    """
    :param y_true: numpy (n_sample,feature_num)
    :param y_pred: numpy (n_sample,feature_num)
    :param mask: numpy (n_sample,feature_num)
    :return: int
    """
    masked_true = []
    masked_pred = []
    for i in range(len(mask)):
        if (mask[i]==0):
            masked_true.extend([y_true[i]])
            masked_pred.extend([y_pred[i]])
        else:
            continue
    mape = masked_mape_np(masked_true, masked_pred,0)

    return mape


def compute_val_loss(net, val_loader, criterion, sw, epoch,DEVICE):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels,mask,timestemps,delta = batch_data
            #print("coeffs.size()",coeffs.size())

            encoder_inputs = encoder_inputs.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)  # (B，N，T，1)
            mask  = mask.unsqueeze(-1).to(DEVICE)
            timestemps = timestemps.unsqueeze(-1).to(DEVICE)
            delta  = delta.unsqueeze(-1).to(DEVICE)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs,mask,timestemps,delta)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]
            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)

               # true_output = labels[:, :, step, :].unsqueeze(-1)
               # pred_mask = mask[:, :, step, :].unsqueeze(-1)
               # predict_output = pred_mask * true_output + (1 - pred_mask) * predict_output

                decoder_input_list = [decoder_start_inputs, predict_output]

            loss,_ = criterion(predict_output, labels, mask)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, data_mask_tensor, epoch, _max, _min, params_path, type,DEVICE):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()
        data_mask_tensor = data_mask_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # 存储所有batch的input

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels,mask,timestemps,delta = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)
            mask = mask.unsqueeze(-1).to(DEVICE)
            timestemps = timestemps.unsqueeze(-1).to(DEVICE)
            delta = delta.unsqueeze(-1).to(DEVICE)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs,mask,timestemps,delta)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]

            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)

               # true_output = labels[:, :, step, :].unsqueeze(-1)
               # pred_mask = mask[:, :, step, :].unsqueeze(-1)
               # predict_output = pred_mask * true_output + (1 - pred_mask) * predict_output

                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mask_MAE(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1),data_mask_tensor.reshape(-1, 1))
        rmse = mask_RMSE(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1),data_mask_tensor.reshape(-1, 1))
        mape = mask_MAPE(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), data_mask_tensor.reshape(-1, 1))
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def get_coeffs(timestamp,x,mask):
    timestamp = timestamp.to(torch.float32)
    new_time = timestamp
    batch_size,num_of_node,num_of_fear,length = x.size()
    for i in range(12-1):
        new_time = torch.cat([new_time,timestamp - i-1],dim=-1)
    new_time = new_time.unsqueeze(1).unsqueeze(1).repeat(1,307,num_of_fear,1).permute(0,1,3,2)

    #### 这里差用nan 替代0
    ######
    #print("start")
    new_mask = mask.unsqueeze(2).bool()
    n = float('nan')

    mask_mx = torch.ones(x.size()) * n
    new_x = torch.where(new_mask,x,mask_mx)
    ######
    #####
    mask = mask.to(torch.float32).unsqueeze(2).permute(0, 1, 3, 2)
    #x = x.to(torch.float32).permute(0, 1, 3, 2)
    new_x = new_x.to(torch.float32).permute(0, 1, 3, 2)
    #print("end")

    '''
    print(new_time.size())
    print(x.size())
    print(mask.size())
    '''


    cde_input = torch.cat([new_time,new_x,mask],dim=-1)
    #print(cde_input.size())
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_input)
    return coeffs

def load_graphdata_normY_channel1(all_data, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
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

    file_data = all_data
    train_x1 = file_data['train_x1']  # (10181, 307, 3, 12)
    train_x1 = train_x1[:, :, 0:1, :]
    train_x2 = file_data['train_x2']  # (10181, 307, 3, 12)
    train_x2 = train_x2[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_mask = file_data['train_mask'] # (10181, 307, 3, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)
    #train_coeffs = file_data['train_coeffs'] # (10181, 307, 11, 36)
    train_delta1 = file_data['train_delta1']
    train_delta2 = file_data['train_delta2']

    train_x_length = train_x1.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x1 = train_x1[:scale]
    train_x2 = train_x2[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]
    #train_coeffs = train_coeffs[:scale]
    train_delta1 = train_delta1[:scale]
    train_delta2 = train_delta2[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_mask = file_data['val_mask']
    val_timestamp = file_data['val_timestamp']
    #val_coeffs = file_data['val_coeffs']  # (10181, 307, 11, 36)
    val_delta = file_data['val_delta']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_mask = file_data['test_mask']
    test_timestamp = file_data['test_timestamp']
    #test_coeffs = file_data['test_coeffs']
    test_delta = file_data['test_delta']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x1[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input1 = np.concatenate((train_decoder_input_start, np.squeeze(train_x1,2)[:, :, :-1]), axis=2)  # (B, N, T)



    train_decoder_input_start = train_x2[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input2 = np.concatenate((train_decoder_input_start, np.squeeze(train_x2, 2)[:, :, :-1]),axis=2)  # (B, N, T)

    train_x1_tensor = torch.from_numpy(train_x1).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    train_x2_tensor = torch.from_numpy(train_x2).type(torch.FloatTensor)  # .to(DEVICE)  # (B, N, F, T)
    train_decoder_input1_tensor = torch.from_numpy(train_decoder_input1).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    train_decoder_input2_tensor = torch.from_numpy(train_decoder_input2).type(torch.FloatTensor)  # .to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    train_mask_tensor = torch.from_numpy(train_mask).type(torch.IntTensor)#.to(DEVICE)
    train_timestamp_tensor = torch.from_numpy(train_timestamp).type(torch.FloatTensor)#.to(DEVICE)
    train_delta1_tensor = torch.from_numpy(train_delta1).type(torch.FloatTensor)
    train_delta2_tensor = torch.from_numpy(train_delta2).type(torch.FloatTensor)

    '''
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(train_timestamp_tensor.size())
    print(train_x_tensor.size())
    print(train_mask_tensor.size())
    train_coeffs_tensor = get_coeffs(train_timestamp_tensor,train_x_tensor,train_mask_tensor)
    print(train_coeffs_tensor.size())
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    '''

    #train_coeffs_tensor = get_coeffs(train_timestamp_tensor, train_x_tensor, train_mask_tensor)

    train_dataset = torch.utils.data.TensorDataset(train_x1_tensor,train_x2_tensor, train_decoder_input1_tensor,train_decoder_input2_tensor, train_target_tensor, train_mask_tensor,train_timestamp_tensor,train_delta1_tensor,train_delta2_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, np.squeeze(val_x,2)[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    val_mask_tensor = torch.from_numpy(val_mask).type(torch.IntTensor)#.to(DEVICE)
    val_timestamp_tensor = torch.from_numpy(val_timestamp).type(torch.FloatTensor)#.to(DEVICE)
    val_delta_tensor = torch.from_numpy(val_delta).type(torch.FloatTensor)
    #val_coeffs_tensor = get_coeffs(val_timestamp_tensor, val_x_tensor, val_mask_tensor)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor,val_mask_tensor,val_timestamp_tensor,val_delta_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, np.squeeze(test_x,2)[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    test_mask_tensor = torch.from_numpy(test_mask).type(torch.IntTensor)#.to(DEVICE)
    test_timestamp_tensor = torch.from_numpy(test_timestamp).type(torch.FloatTensor)#.to(DEVICE)
    test_delta_tensor = torch.from_numpy(test_delta).type(torch.FloatTensor)
    #test_coeffs_tensor = get_coeffs(test_timestamp_tensor, test_x_tensor, test_mask_tensor)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor,test_mask_tensor,test_timestamp_tensor,test_delta_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x1_tensor.size(), train_decoder_input1_tensor.size(), train_target_tensor.size(),train_mask_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size(),val_mask_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size(),test_mask_tensor.size())

    return train_loader, train_target_tensor,train_mask_tensor, val_loader, val_target_tensor,val_mask_tensor, test_loader, test_target_tensor,test_mask_tensor, _max, _min

