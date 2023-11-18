import numpy as np
import random
import csv
import os
import pandas as pd
from communities.algorithms import louvain_method
import argparse
import configparser


class prepare_miss_data:
    def __init__(self, dataPath, savedatapath, distancePath, miss_rate, miss_func, patch_size):              
        self.dataPath = dataPath
        self.savedatapath = savedatapath
        self.distancePath = distancePath
        self.miss_rate = miss_rate
        self.miss_func = miss_func
        self.patch_size = patch_size
    
    def load_data(self):  
        '''
        load data from .npz file
        :return: original data
        '''
        if os.path.isdir(self.dataPath): #os.path.isdir():判断某一对象(需提供绝对路径)是否为目录
            files = os.listdir(self.dataPath)  #os.listdir():返回一个列表，其中包含有指定路径下的目录和文件的名称
            data = []
            for file in files:
                position = os.path.join(self.dataPath, file)
                print("processing:", position)
                X = np.load(position)['data']
                data.append(X)
            data = np.concatenate(data, 0)
        else:   #执行这一分支
            data = np.load(self.dataPath)['data']   #从pems04.npz中提取出类别为data的数据
            print('data shape:',data.shape)
      
        return data
    

    def get_graph_classes(self,distance_of_Path, num_of_vertices, id_filename=None):
        '''
        :Param distancePath: path to the file where distances between nodes are stored
        :Param num_of_vertices: the number of nodes
        :Param id_filename:
        :return: communities, list, community based on distances between nodes
        '''
        if '.npy' in distance_of_Path:          
            adj_mx = np.load(distance_of_Path)
            return adj_mx, None

        else:
            A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                         dtype=np.float32)
            distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
            if id_filename:
                with open(id_filename, 'r') as f:
                    id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
   
                with open(distance_of_Path, 'r') as f:
                    f.readline()  # 略过表头那一行
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])
                        A[id_dict[i], id_dict[j]] = 1
                        A[id_dict[j], id_dict[i]] = 1
                        distaneA[id_dict[i], id_dict[j]] = distance
                        distaneA[id_dict[j], id_dict[i]] = distance

            else:  # distance file中的id直接从0开始
                with open(distance_of_Path, 'r') as f:
                    f.readline()
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])    #类似建立一个无向图？
                        A[i, j] = 1
                        A[j, i] = 1
                        distaneA[i, j] = distance
                        distaneA[j, i] = distance
        communities, _ = louvain_method(A)          #Louvain 算法对最大化图模块性的社区进行贪婪搜索。如果一个图具有高密度的群体内边缘和低密度的群体间边缘，则称之为模图。
        return communities


    def mask_and_save_data(self, ori_data, miss_func, miss_rate, mask):         
        '''
        storage data and mask, two files with true data or miss data and mask
        :Param ori_data: original data, shape(16992,307,3)
        :Param miss_func: missing function, str
        :Param miss_rate: missing rate, float
        :Param mask: mask in missing function, shape(16992,307,3)
        '''

        true_data_savename = 'true_data_' + miss_func + '_' +str(miss_rate)+ '_v2' +'.npz'
        save_path = os.path.join(self.savedatapath, true_data_savename)
        data = ori_data[:, :, :]
        np.savez_compressed(save_path, data=data, mask=mask)
        print('missing rate', miss_rate, miss_func, 'true data and mask save in:',save_path)

        miss_data_savename = 'miss_data_' + miss_func + '_' +str(miss_rate)+ '_v2' +'.npz'
        save_path = os.path.join(self.savedatapath, miss_data_savename)
        miss_data = np.where(mask==1, ori_data, 0)
        np.savez_compressed(save_path, data=miss_data, mask=mask)
        print('missing rate',miss_rate, miss_func, 'miss data and mask save in:',save_path)


    def mask_array(self, ori_data, miss_rate, miss_func, patch_size): 
        '''
        :Param ori_data: original data, shape(16992,307,3)
        :Param miss_rate: missing rate, float
        :Param miss_func: missing function, str
        :Param patch_size:
        :return: mask, shape(16992,307,3)
        '''  
        if miss_func not in ['SR-TR','SC-TR','SR-TC','SC-TC']:
            print("miss_func type error!please select from  'SR-TR','SC-TR','SR-TC','SC-TC'")
            return -1
        else:
            T, N, F = ori_data.shape
            num_of_node = N
            mask = np. ones_like(ori_data)

            if miss_func == 'SR-TR':               

                "get new mask" 
                rm = np.random.rand(T,N,1) 
                rm = np.where(rm <= miss_rate, 0, 1)
                mask = np.where(rm == 1, mask, rm)

                "mask and save data"
                self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)

                return mask

            if miss_func == 'SR-TC':              

                "get new mask" 
                rm = np.random.rand(round(T/patch_size),N,1)               
                rm = np.where(rm <= miss_rate, 0, 1)
                rm = rm.repeat(patch_size, axis=0) 
                mask = np.where(rm == 1, mask, rm)

                "mask and save data"
                self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)

                return mask
            
            if miss_func == 'SC-TR':

                "get communities"
                communities = self.get_graph_classes(self.distancePath,num_of_node)
                communities = [list(communities[i]) for i in range(len(communities))]

                "####new save communities"
                community_save_name = 'communities of PEMS04' + '.npz'
                community_save_path = os.path.join(self.savedatapath,community_save_name)
                np.savez_compressed(community_save_path, length = len(communities), communities=communities)

                "get new mask"
                num_cluster = len(communities)
                rm = np.random.rand(T, num_cluster)
                x = round(T*N*miss_rate/num_of_node)    #四舍五入，最终得到的miss rate会有少许偏差
                for j in range(num_cluster):              
                    for i in range(T):
                        if rm[i][j] <= x/T:
                            t, n = i, j
                            mask_node = communities[n]
                            mask[t, mask_node, :] = 0    #mask各社区中的node在t时刻的数据
               
                "mask and save data"
                self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)

                return mask
            
            if miss_func == 'SC-TC':

                "get communities"
                communities = self.get_graph_classes(self.distancePath,num_of_node)
                communities = [list(communities[i]) for i in range(len(communities))]

                "get new mask"
                num_cluster = len(communities)
                rm = np.random.rand(round(T/patch_size), num_cluster)
                x = round(T * N * miss_rate / (num_of_node * patch_size))
                for j in range(num_cluster):
                    for i in range(round(T/patch_size)):
                        if rm[i][j] <= x * patch_size / T:
                            t,n = i,j
                            start = t * patch_size
                            end = (t + 1) * patch_size
                            mask[start:end,communities[n],:] = 0
                
                "mask and save data"
                self.mask_and_save_data(ori_data, miss_func, miss_rate, mask)               

                return mask
     


    def generate_miss_data(self):
        data = self.load_data()  #load.. return data
        mask = self.mask_array(data, self.miss_rate, self.miss_func, self.patch_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default='configurations/PEMS04.conf', type=str,
                    help="configuration file path")
  parser.add_argument("--patch", default=16, type=int,
                    help="time patch size")
  parser.add_argument("--missrate", default=0.2, type=float,
                    help="missrate")
  parser.add_argument("--misstype", default="SC-TR", type=str,
                    help="misstype: 'SR-TR','SC-TR','SR-TC','SC-TC' ")
  args = parser.parse_args()
  config = configparser.ConfigParser()
  print('Read configuration file: %s' % (args.config))

  config.read(args.config)
  config = config["generator"]
  graph_signal_matrix_filename = config["graph_signal_matrix_filename"]
  save_filesdir = config["save_filesdir"]
  distancePath = config["distancePath"]

  patch_size = args.patch
  missrate = args.missrate
  misstype = args.misstype
      
  a=prepare_miss_data(graph_signal_matrix_filename, save_filesdir, distancePath, missrate,misstype, patch_size)
  a.generate_miss_data()