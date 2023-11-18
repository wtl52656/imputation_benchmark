import numpy as np
import torch
import os

miss_mode='NM'
miss_rate = 0.2

#basepath = './' + miss_mode + '/miss_rate=' + str(miss_rate)

basepath = './datasets'
filename = 'true_data.npz'
data = np.load(os.path.join(basepath,filename))
true_data = data['data']
old_mask = data['mask']
true_data = torch.from_numpy(true_data).to(torch.float32)
old_mask = torch.from_numpy(old_mask)
new_mask = old_mask[:,:,0].unsqueeze(-1).repeat(1,1,old_mask.shape[-1])

### revise true data
true_data_savename = 'true_data_' + miss_mode + '_' +str(miss_rate) +'.npz'
np.savez_compressed(os.path.join(basepath,true_data_savename), data=true_data.numpy(),mask=new_mask.numpy())
print("end save revised true data!")

##### get miss data
miss_data_savename = 'miss_data_' + miss_mode + '_' +str(miss_rate) +'.npz'

n = 0
mask_mx = torch.ones(true_data.size()) * n
miss_data = torch.where(new_mask==1,true_data,mask_mx)
np.savez_compressed(os.path.join(basepath,miss_data_savename), data=miss_data.numpy(),mask=new_mask.numpy())
print("end save miss  data!")


