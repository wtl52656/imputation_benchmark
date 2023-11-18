from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd
import random
import os


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)|(y_true > 1e-1 )
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                    y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def mask_arrray(data, loss_ratio, seed):
    random.seed(seed)
    mask = [1 for i in range(data.size)]
    for i in range(int(data.size * loss_ratio)):
        mask[i] = 0
    random.shuffle(mask)
    mask = np.array(mask).reshape(data.shape)
    return mask

def mask_MAE(y_true,y_pred,mask):
    """

    :param y_true:numpy (n_sample,feature_num)
    :param y_pred:numpy (n_sample,feature_num)
    :param mask:numpy (n_sample,feature_num)
    :return: int
    """
    masked_true = []
    masked_pred = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0: # 人为挖空值
                masked_true.extend([y_true[i][j]])
                masked_pred.extend([y_pred[i][j]])
    mae = mean_absolute_error(masked_true,masked_pred)

    # masked_true = np.array(masked_true)
    # masked_pred = np.array(masked_pred)
    # diff = abs(masked_true - masked_pred)

    # print("len: ", len(diff))
    # print("max mae: ", max(diff))
    # print("min mae: ", min(diff))

    # data = np.stack([masked_true, masked_pred, diff], axis=1)
    # data = pd.DataFrame(data)
    # data.columns = ['true', 'prediction', 'diff']
    # data.to_excel("for_check.xlsx", float_format="%.4f") # 结果存入excel文件中，利用wps进行可视化

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
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                masked_true.extend([y_true[i][j]])
                masked_pred.extend([y_pred[i][j]])
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
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                masked_true.extend([y_true[i][j]])
                masked_pred.extend([y_pred[i][j]])
    #mape = mean_absolute_percentage_error(masked_true, masked_pred)
    mape = masked_mape_np(masked_true, masked_pred,0.0)

    return mape

def calculate_metric(true_data,pred_data,mask,dim):
    """
    :param true_data: numpy (time_step,node_num,feature_num)
    :param pred_data: numpy (time_step,node_num,feature_num)
    :param mask: numpy
    :param dim: int
    :return:
    """
    true_data = true_data[:,:,dim]
    pred_data =pred_data[:,:,dim]
    mask = mask[:,:,dim]

    print(true_data.shape)
    print(pred_data.shape)
    print(mask.shape)
    mae = mask_MAE(true_data,pred_data,mask)
    rmse = mask_RMSE(true_data,pred_data,mask)
    mape = mask_MAPE(true_data,pred_data,mask)

    return mae,rmse,mape

def get_true_data(datapath):
    true_data_file_name = 'true_data_0.3.npz'
    true_data_path = os.path.join(datapath,true_data_file_name)
    true_dataset = np.load(true_data_path)
    true_data = true_dataset['data']
    mask = true_dataset['mask']
    #print(true_data.shape)
    #print(mask.shape)
    return true_data,mask

def get_pred_data(datapath,datadir):
    pred_data_path = os.path.join(datapath,datadir)
    files = os.listdir(pred_data_path)
    data = []
    for file in files:
        position = os.path.join(pred_data_path, file)
        print("processing:", position)
        X = np.load(position,allow_pickle=True)['data']
        data.append(X)
    data = np.concatenate(data, 0)

    return data

def get_impute_data(datapath, datadir):
    data = np.load(os.path.join(datapath, datadir))
    return data['impute'], data['true'], data['mask']

if __name__ == "__main__":
    data_path = 'result/nm'  # 有问题
    #datadir = 'e2egan'
    #datadir = 'gan2stage'
    #datadir = 'last'
    #datadir = 'birts'
    datadir = 'pems_miss_ratio_0.3.npz'
    true_data,mask = get_true_data(data_path)

    print("true_data: ", true_data.shape, true_data)
    print("mask: ", mask.shape, mask)

    pred_data, mask_data, mask_ = get_impute_data(data_path,datadir)
    print("pred_data:", pred_data.shape, pred_data)
    print("mask_data",  mask_data.shape, mask_data)
    print("check mask: ", mask_.shape, mask_)
    mae,rmse,mape = calculate_metric(true_data,pred_data,mask,0)
    print("mae,rmse,mape:%.4f %.4f %.4f," % (mae,rmse,mape))
