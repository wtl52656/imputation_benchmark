import numpy as np
import torch
import os
basepath = 'datasets/'
filename = 'true_data.npz'


savename = 'miss_data.npz'
data = np.load(os.path.join(basepath,filename))
true_data = data['data']
mask = data['mask']

true_data = torch.from_numpy(true_data).to(torch.float32)
mask = torch.from_numpy(mask)
n = 0 # 缺失值填充为0
mask_mx = torch.ones(true_data.size()) * n
print(true_data.size())
print(mask_mx.size())


miss_data = torch.where(mask==1,true_data,mask_mx)
print(miss_data.numpy().shape)
print(miss_data)
np.savez_compressed(os.path.join(basepath,savename), data=miss_data.numpy(),mask=mask.numpy())


