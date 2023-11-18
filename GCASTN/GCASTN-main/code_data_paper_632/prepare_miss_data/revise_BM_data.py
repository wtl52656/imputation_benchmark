import numpy as np
import torch
import os
import csv
from communities.algorithms import louvain_method

distance_df_filename = './PEMS04.csv'
num_of_vertices = 307

def get_graph_classes(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

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

            with open(distance_df_filename, 'r') as f:
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
    communities, _ = louvain_method(A)
    return communities

get_graph_classes(distance_df_filename, num_of_vertices)

def get_miss_start_and_end(mask):
    starts = []
    ends = []
    ancher= 1
    for i in range(mask.shape[0]):
        if mask[i][0][0]==0:
            if ancher == 1:
                start = i
                starts.append(start)
                ancher = 0
        if mask[i][0][0]==1:
            if ancher == 0:
                end = i
                ends.append(end)
                ancher = 1
    if len(ends)<len(starts):
        ends.append(mask.shape[0] -1)
    return starts,ends

def get_new_mask(mask,block_window,miss_rate,communities):
    communities = [list(communities[i]) for i in range(len(communities))]
    dim2 = len(communities)
    dim1 = int(mask.shape[0] / block_window)

    vec = np.round(np.random.rand(dim1, dim2) + 0.5 - miss_rate)
    for i in range(dim1):
        for j in range(dim2):
            if vec[i][j]==0:
                start = i * block_window
                end = (i + 1) * block_window
                ob_node = communities[j]
                mask[start:end + 1, ob_node, :] = 0


    return mask






miss_mode='BM'
miss_rate = 0.2
block_window = 6

#basepath = './' + miss_mode + '/miss_rate=' + str(miss_rate)

basepath = './datasets'
filename = 'true_data.npz'
data = np.load(os.path.join(basepath,filename))
true_data = data['data']
old_mask = data['mask']
true_data = torch.from_numpy(true_data).to(torch.float32)
old_mask = torch.from_numpy(old_mask)

##### get new mask
communities = get_graph_classes(distance_df_filename, num_of_vertices)

new_mask = get_new_mask(old_mask,block_window,miss_rate,communities)



### revise true data
true_data_savename = 'true_data_' + miss_mode + '_' +str(miss_rate)+ '_v2' +'.npz'
np.savez_compressed(os.path.join(basepath,true_data_savename), data=true_data.numpy(),mask=new_mask.numpy())
print("end save revised true data!")

##### get miss data
miss_data_savename = 'miss_data_' + miss_mode + '_' +str(miss_rate)+ '_v2' +'.npz'

n = 0
mask_mx = torch.ones(true_data.size()) * n
miss_data = torch.where(new_mask==1,true_data,mask_mx)
np.savez_compressed(os.path.join(basepath,miss_data_savename), data=miss_data.numpy(),mask=new_mask.numpy())
print("end save miss  data!")
a = miss_data.numpy()
print("--")


