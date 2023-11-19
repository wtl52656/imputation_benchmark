import os
import pandas as pd
import numpy as np
import ujson as json
from tqdm import tqdm
import prepareData
import argparse
from pathlib import Path
import configparser
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default=None)
parser.add_argument("--config", default='configurations/PEMS04_prepare.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()


config = configparser.ConfigParser()
config.read(args.config)
config = config["prepare"]

fs_train = None
fs_test = None

seq_len = int(config["seq_len"]) # 一个小时
attributes = int(config["attributes"]) # 特征维度 第一个维度 流量
fill_vale = int(config["fill_value"]) # 填充值
test_ratio = float(config["test_ratio"]) #测试集比例
val_ratio = float(config["val_ratio"]) #验证集比例
file_prefix = config["file_prefix"]

mean = np.zeros((attributes,))
std = np.zeros((attributes,))

def parse_delta(masks, dir_):  # time gaps

    deltas = []

    for time_step in range(0, seq_len):  # 递推
        if time_step == 0:
            deltas.append(np.zeros(Nattributes))  # 初始值与论文中的不一致！-> 修正
        else:
            deltas.append(np.ones(Nattributes) + (1 - masks[time_step]) * deltas[-1])  # 时间间隔一样，都为1，即s(t)-s(t-1)=1
    # print("deltas: ", deltas)

    return np.array(deltas)


def parse_rec_test(values, masks, true_data, dir_):
    deltas = parse_delta(masks, dir_)

    rec = {}

    rec['values'] = np.nan_to_num(values).astype("float64").tolist() # nan -> 0
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['true_data'] = true_data.astype("float64").tolist()
    rec['deltas'] = deltas.astype("int32").tolist()
    # print("rec: ", rec)

    return rec

def parse_rec_train(values, masks, dir_):
    deltas = parse_delta(masks, dir_)

    rec = {}

    rec['values'] = np.nan_to_num(values).astype("float64").tolist() # nan -> 0
    rec['masks'] = masks.astype('int32').tolist()

    rec['deltas'] = deltas.astype("int32").tolist()
    # print("rec: ", rec)
    
    return rec

def parse_sample(values,masks,true_data, sample_type):
    if masks.sum() == 0: # 样本中数据全部缺失！ BM模式过滤很多样本！
        return

    rec = {}

    if sample_type == 'train':
        # prepare the model for both directions
        rec['forward'] = parse_rec_train(values, masks, dir_='forward')
        rec['backward'] = parse_rec_train(values[::-1], masks[::-1], dir_='backward')

        rec = json.dumps(rec)

        fs_train.write(rec + '\n')


    elif sample_type == 'test':
        rec['forward'] = parse_rec_test(values,masks, true_data=true_data, dir_='forward')
        rec['backward'] = parse_rec_test(values[::-1], masks[::-1], true_data=true_data[::-1], dir_='backward')

        rec = json.dumps(rec)
        
        fs_test.write(rec + "\n")

    
    elif sample_type == 'val':
        rec['forward'] = parse_rec_test(values,masks, true_data=true_data, dir_='forward')
        rec['backward'] = parse_rec_test(values[::-1], masks[::-1], true_data=true_data[::-1], dir_='backward')

        rec = json.dumps(rec)
        
        fs_val.write(rec + "\n")


flag = 0

def parse_train(sample):
    global flag

    mask = sample['masks'] 

    values = sample['values'] 
    
    if flag == 0:
        flag = 1
        print("train sample's shape: ", values[:, :, :attributes].shape)

    parse_sample(  np.reshape(values[:, :, :attributes],(values.shape[0],-1)),\
                 np.reshape(mask[:, :, :attributes],(mask.shape[0],-1)), None,"train")


def parse_test(sample):
    global flag

    values = sample['values'] # 100%可观测值
    true_data = sample['true_data'] # 完整数据集
    mask = sample['masks'] # 100%可观测值的mask

    if flag == 0:
        flag = 1 # 只输出一次
        print("test sample's shape: ", values[:, :, :attributes].shape)


    parse_sample(np.reshape(values[:, :, :attributes],(values.shape[0],-1)),\
                 np.reshape(mask[:, :, :attributes],(mask.shape[0],-1)), np.reshape(true_data[:, :, :attributes],(true_data.shape[0],-1)),"test")


def parse_val(sample):
    global flag

    values = sample['values'] # 100%可观测值
    true_data = sample['true_data'] # 完整数据集
    mask = sample['masks'] # 100%可观测值的mask

    if flag == 0:
        flag = 1 # 只输出一次
        print("val sample's shape: ", values[:, :, :attributes].shape)

    parse_sample(np.reshape(values[:, :, :attributes],(values.shape[0],-1)),\
                 np.reshape(mask[:, :, :attributes],(mask.shape[0],-1)), np.reshape(true_data[:, :, :attributes],(true_data.shape[0],-1)),"val")


if __name__ == "__main__":


    print(f"***{config['type']} {config['miss_rate']} is starting ...***")

    true_data_path = os.path.join(config['ori_file_prefix'],f"true_data_{config['type']}_{config['miss_rate']}_v2.npz")
    miss_data_path = os.path.join(config['ori_file_prefix'],f"miss_data_{config['type']}_{config['miss_rate']}_v2.npz")

    true_data = np.load(true_data_path, allow_pickle=True)['data'][:,:,:attributes]
    mask = np.load(true_data_path, allow_pickle=True)['mask'][:,:,:attributes]
    miss_data = np.load(miss_data_path, allow_pickle=True)['data'][:,:,:attributes]
    print(true_data.shape)

    Nattributes = true_data.shape[1]*attributes

    print("true_data: ", true_data.shape, ", mask: ", mask.shape)

    # 可观测值
    for i in range(attributes):
        mean[i] = np.mean(true_data[mask[..., i]==1, i])
        std[i] = np.std(true_data[mask[..., i]==1, i])
        
    
    
    print(f"mean: {mean}, std: {std}")
    fs_mean_std = open(os.path.join(file_prefix,f"data_meanstd_{config['type']}_{config['miss_rate']}.pkl"),"wb")
    pickle.dump({'mean':mean, 'std':std}, fs_mean_std)

    miss_data = np.where(mask == 1, miss_data, np.nan)
    miss_data = (miss_data - mean) / std # 标准化（广播）

    values = miss_data  # 后续接口用到名称都是values，所以此处也用values

    train_file_path = os.path.join(file_prefix,f"data_train_{config['type']}_{config['miss_rate']}.json")
    test_file_path = os.path.join(file_prefix,f"data_test_{config['type']}_{config['miss_rate']}.json")
    val_file_path = os.path.join(file_prefix,f"data_val_{config['type']}_{config['miss_rate']}.json")

    fs_train = open(train_file_path, 'w')
    fs_test = open(test_file_path, 'w')
    fs_val = open(val_file_path, 'w')

    train, test , val = prepareData.generate_samples_brits(true_data, mask,  values,  seq_len,test_ratio,val_ratio) # 从训练集中划分出验证集！！！
    
    flag = 0

    for sample in tqdm(test):
        parse_test(sample)

    # exit()
    flag = 0

    for sample in tqdm(train):

        parse_train(sample)

    flag = 0
    
    for sample in tqdm(val):
        parse_val(sample)
    

    print(f"***{config['type']} {config['miss_rate']} is ending ...***")