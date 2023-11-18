import numpy as np
import random
import time
import csv
import os
import pandas as pd

graph_signal_matrix_filename = './PEMS04/pems04.npz' # 无缺失的数据集原始文件
topopath = './PEMS04/distance.csv' # 数据集所需要的节点距离文件
save_filesdir = './datasets' # 生成数据文件存放文件夹


class Pemsdata_preprocess:
    def __init__(self, dataPath, savedatapath, topopath, sample_len, day_slice_len, miss_rate,miss_func, is_filter_node=False,
                is_filter_time=False, delete_day=0):
        self.dataPath = dataPath
        self.savedatapath = savedatapath
        self.topopath = topopath
        self.sample_len = sample_len
        self.day_slice_len = day_slice_len
        self.miss_rate = miss_rate
        self.miss_func = miss_func
        self.is_filter_node = is_filter_node
        self.is_filter_time = is_filter_time
        self.delete_day = delete_day

    def filter_node(self, ratio=0.85):
        data = pd.read_csv(self.topopath)
        save_filter_disfilenname = 'filter_distance.csv'
        save_filter_idfilename = 'filter_307.txt'
        dirpath = os.path.dirname(self.topopath)
        # print(dirpath)
        save_filter_distance_path = os.path.join(dirpath, save_filter_disfilenname)
        save_filter_id_path = os.path.join(dirpath, save_filter_idfilename)

        random.seed(2)
        loss_ratio = ratio
        mask = [1 for i in range(data.shape[0])]
        for i in range(int(data.shape[0] * loss_ratio)):
            mask[i] = 0
        random.shuffle(mask)
        # print("mask",mask)
        csv_reader = csv.reader(open(self.topopath))
        row_num = 0
        c = open(save_filter_distance_path, "w", newline='')
        writer = csv.writer(c)
        writer.writerow(['from', 'to', 'cost'])
        row_id = 0
        nodeout = set()
        nodein = set()

        for row in csv_reader:
            if row_num == 0:
                row_num += 1
                continue
            else:
                if mask[row_id] == 1:
                    writer.writerow([row[0], row[1], row[2]])
                    nodeout.add(int(row[0]))
                    nodein.add(int(row[1]))
                    row_id += 1
                else:
                    row_id += 1
                    continue

        c.close()

        filternode = nodeout | nodein
        # print(len(filternode))
        with open(save_filter_id_path, 'w', encoding='utf-8') as f:
            for i in range(len(filternode)):
                text = str(i) + '\n'
                f.write(text)
        f.close()

        return list(filternode)

    def load_and_filter_data(self, is_filter_node=False, is_filter_time=False, delete_day=0):
        if os.path.isdir(self.dataPath):
            files = os.listdir(self.dataPath)
            data = []
            for file in files:
                position = os.path.join(self.dataPath, file)
                print("processing:", position)
                X = np.load(position)['data']
                data.append(X)
            data = np.concatenate(data, 0)
        else:
            data = np.load(self.dataPath)['data']
            print(data.shape)

        if is_filter_node:
            print("filter node....")
            filter_node = sorted(self.filter_node())
            # print(filter_node)
            # print(data.shape)
            data = data[:, filter_node, :]
            # print(data.shape)
        if is_filter_time:
            print("filter time....")
            if delete_day != 0:
                time_stemp = delete_day * self.day_slice_len
                data = data[:-time_stemp, :, :]

        return data

    def get_mask(self,data):
        mask = [1 for i in range(data.size)]
        mask = np.array(mask).reshape(data.shape)
        # print(mask.shape)
        true_mask = np.where(data==0,data,mask)
        return true_mask

    def ten2mat(self,tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

    def mat2ten(self,mat, dim, mode):
        index = list()
        index.append(mode)
        for i in range(dim.shape[0]):
            if i != mode:
                index.append(i)
        return np.moveaxis(np.reshape(mat, list(dim[index]), order='F'), 0, mode)


    def mask_arrray(self, data, loss_ratio,miss_func, seed):
        if miss_func not in ['RM','NM','BM']:
            print("miss_func type error!please select from  'RM','NM','BM'")
            return -1
        else:
            if miss_func == 'RM':
                dim_1, dim_2, dim_3 = data.shape
                total_mask = []
                for i in range(data.shape[-1]):
                    data_feature = data[:, :, i].reshape(-1, self.day_slice_len, dim_2).transpose(2, 1, 0) + 4
                    dim1,dim2,dim3 = data_feature.shape
                    mask_data = data_feature * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - loss_ratio)
                    mask = self.get_mask(mask_data)
                    total_mask.append(mask)
                total_mask = np.array(total_mask).transpose(1,2,3,0).reshape(dim_2,dim_1,dim_3).transpose(1, 0, 2)
                return total_mask

            if miss_func == 'NM':
                dim_1, dim_2, dim_3 = data.shape
                total_mask = []
                for i in range(data.shape[-1]):
                    data_feature = data[:, :, i].reshape(-1, self.day_slice_len, dim_2).transpose(2, 1, 0) + 4
                    dim1, dim2, dim3 = data_feature.shape
                    mask_data = data_feature * np.round(np.random.rand(dim1, dim3) + 0.5 - loss_ratio)[:, None, :]
                    mask = self.get_mask(mask_data)
                    total_mask.append(mask)
                total_mask = np.array(total_mask).transpose(1,2,3,0).reshape(dim_2,dim_1,dim_3).transpose(1, 0, 2)
                return total_mask
            else:
                dim_1, dim_2, dim_3 = data.shape
                total_mask = []
                for i in range(data.shape[-1]):
                    data_feature = data[:, :, i].reshape(-1, self.day_slice_len, dim_2).transpose(2, 1, 0) + 4
                    dim1, dim2, dim3 = data_feature.shape
                    dim_time = dim2 * dim3
                    block_window = 6
                    vec = np.random.rand(int(dim_time / block_window))
                    temp = np.array([vec] * block_window)
                    vec = temp.reshape([dim2 * dim3], order='F')
                    mask_data = self.mat2ten(self.ten2mat(data_feature, 0) * np.round(vec + 0.5 - loss_ratio)[None, :],np.array([dim1, dim2, dim3]), 0)
                    mask = self.get_mask(mask_data)
                    total_mask.append(mask)
                total_mask = np.array(total_mask).transpose(1,2,3,0).reshape(dim_2,dim_1,dim_3).transpose(1, 0, 2)
                return total_mask

    def mask_data(self):
        data = self.load_and_filter_data(self.is_filter_node, self.is_filter_time, self.delete_day)
        mask = self.mask_arrray(data, self.miss_rate,self.miss_func, 2)
        print(data.shape)
        #print(mask)
        save_filename = 'true_data.npz'
        dirpath = self.savedatapath
        save_path = os.path.join(dirpath, save_filename)
        print("saving the true data and mask:", save_path)
        data = data[:, :, :3]
        np.savez_compressed(save_path, data=data, mask=mask)

        return data, mask

class Genaratepemsinputationdata:
    def __init__(self, dataPath, savedatapath, topopath, sample_len, day_slice_len, miss_rate,miss_func, is_filter_node=False,
                is_filter_time=False, delete_day=0):
        self.dataPath = dataPath
        self.savedatapath = savedatapath
        self.topopath = topopath
        self.sample_len = sample_len
        self.day_slice_len = day_slice_len
        self.miss_rate = miss_rate
        self.miss_func = miss_func
        self.is_filter_node = is_filter_node
        self.is_filter_time = is_filter_time
        self.delete_day = delete_day

    def genarate_time(self, seq_No):
        seq_No = seq_No + 1
        day = int(int(seq_No / 12) / 24)
        hour = int(int(seq_No / 12) % 24)
        min = (seq_No % 12) * 5
        time_stemp = str(day).rjust(2, '0') + ':' + str(hour).rjust(2, '0') + ':' + str(min).rjust(2, '0')
        # print(type(time_stemp))
        return time_stemp

    def generate_data_for_imputation(self):
        pems = Pemsdata_preprocess(self.dataPath, self.savedatapath, self.topopath, self.sample_len, self.day_slice_len,
                                self.miss_rate,self.miss_func, self.is_filter_node, self.is_filter_time, self.delete_day)
        data, mask = pems.mask_data()

a = Genaratepemsinputationdata(graph_signal_matrix_filename, save_filesdir, topopath, 576, 288, 0.2,'RM', False, True, 1)

a.generate_data_for_imputation()



