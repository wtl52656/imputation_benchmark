import numpy as np



def get_sample_by_overlaped_Sliding_window_brits(true_data,masks,values,sample_len):
    samples=[]
    for i in range(true_data.shape[0]):
        if i+sample_len>true_data.shape[0]:
            break
        else:
            #  "get a sample"
            sample = dict()
            sample['true_data'] = true_data[i:i+sample_len]
            sample["masks"] = masks[i:i+sample_len]
            sample['values'] = values[i:i+sample_len]
            samples.append(sample)
    return samples

def get_sample_by_unoverlaped_Sliding_window_brits(true_data,masks,values,sample_len):
    samples=[]
    for i in range(int(true_data.shape[0]//sample_len)):
        if i+sample_len>true_data.shape[0]:
            break
        else:
            # "get a sample by indice[i*sample_len,(i+1)*sample_len]"
            sample = dict()
            sample['true_data'] = true_data[i*sample_len:(i+1)*sample_len]
            sample["masks"] = masks[i*sample_len:(i+1)*sample_len]
            sample['values'] = values[i*sample_len:(i+1)*sample_len]
            samples.append(sample)
    return samples


def get_sample_by_overlaped_Sliding_window_only_draw(X, Y,  mask, sample_len):
    #X,Y,mask: shape(T,N,1)
    X_window,Y_window, mask_window = [], [], []
    end_window = X.shape[0]//sample_len
    for i in range(end_window):
        X_window.append(X[i*sample_len:sample_len*(i+1)])
        Y_window.append(Y[i*sample_len:sample_len*(i+1)])
        mask_window.append(mask[i*sample_len:sample_len*(i+1)])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)

    return X_window,Y_window, mask_window

def generate_samples_brits(true_data,masks,values,sample_len,test_ratio,val_ratio):
    data_num = true_data.shape[0]

    test_len = int(data_num * test_ratio)
    # val_end_idx = int(test_end_idx + data_num * val_ratio)
    val_len = int(data_num * val_ratio)

    train_X, val_X, test_X = values[ :-(val_len+test_len)], \
                             values[ -(val_len+test_len):-(test_len)],\
                             values[-test_len:]

    train_Y, val_Y, test_Y = true_data[ :-(val_len+test_len)], \
                             true_data[ -(val_len+test_len):-(test_len)],\
                             true_data[-test_len:]

    train_mask, val_mask, test_mask = masks[ :-(val_len+test_len)], \
                             masks[ -(val_len+test_len):-(test_len)],\
                             masks[-test_len:]
    
    training_set = get_sample_by_overlaped_Sliding_window_brits(train_Y,  train_mask, train_X, sample_len)
    valing_set= get_sample_by_overlaped_Sliding_window_brits(val_Y,  val_mask, val_X, sample_len)
    testing_set = get_sample_by_overlaped_Sliding_window_brits(test_Y, test_mask, test_X, sample_len)
    # print(len(testing_set), testing_set[0])


    # all_samples = get_sample_by_overlaped_Sliding_window_brits(true_data,masks,values,sample_len) # 滑窗
    #all_samples = get_sample_by_unoverlaped_Sliding_window_brits(true_data,masks,values,sample_len) # 不滑窗
    
    import random
    # random.shuffle(all_samples)

    

    # training_set = all_samples[:-(test_end_idx+val_end_idx)] 
    # testing_set = all_samples[-test_end_idx:]
    # valing_set = all_samples[-(val_end_idx+test_end_idx):-(test_end_idx)]

    # training_set = all_samples[val_end_idx:] 
    # testing_set = all_samples[:test_end_idx]
    # valing_set = all_samples[test_end_idx:val_end_idx]

    return training_set, testing_set, valing_set

if __name__ == "__main__":
    pass