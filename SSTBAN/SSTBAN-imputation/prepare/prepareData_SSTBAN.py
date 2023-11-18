import numpy as np
import os
import pandas as pd
import argparse
import configparser
import warnings
import datetime
import re

warnings.filterwarnings('ignore')


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def seq2instance_plus(data, num_his, num_pred):
    num_step = data.shape[0]
    num_sample = num_step - num_his - num_pred + 1
    x = []
    y = []
    for i in range(num_sample):
        x.append(data[i: i + num_his])
        y.append(data[i + num_his: i + num_his + num_pred, :, :1])
    x = np.array(x)
    y = np.array(y)
    return x, y

def SlideWindows(data,time_embedding,condmask,obmask, window=12):

    length = len(data)
    end_index = length  - window + 1

    Datas = [] 
    Cond_masks = []
    Ob_masks = []
    te = []
    index = 0

    while index < end_index:
        Datas.append(data[index:index+window])
        Cond_masks.append(condmask[index:index+window])
        Ob_masks.append(obmask[index:index+window])
        te.append(time_embedding[index:index+window])
        index = index + 1

    Datas = np.asarray(Datas)
    Cond_masks = np.asarray(Cond_masks)
    Ob_masks = np.asarray(Ob_masks)
    te = np.asarray(te)

    return Datas, te, Cond_masks, Ob_masks

def get_randmask( observed_mask):
    rand_for_mask = np.random.randn(*observed_mask.shape) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()  # missing ratio
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        #rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        rand_for_mask[i][rand_for_mask[i].argsort()[-num_masked:]] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape)
    return cond_mask

def get_hist_mask( observed_mask):
    for_pattern_mask = observed_mask
    rand_mask = get_randmask(observed_mask)

    cond_mask = observed_mask.copy()
    for i in range(len(cond_mask)):
        mask_choice = np.random.rand()
        if mask_choice > 0.5:
            cond_mask[i] = rand_mask[i]
        else:  # draw another sample for histmask (i-1 corresponds to another sample)
            cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
    return cond_mask

def loadImputationData(config):
    data_config = config['Data']
    training_config = config['Training']
    time_config = config['Time']

    time_slice_size = int(data_config['time_slice_size'])
    train_ratio = float(data_config['train_ratio'])
    val_ratio = float(data_config['val_ratio'])
    test_ratio = float(data_config['test_ratio'])
    sample_len = int(data_config['sample_len'])
    num_of_vertices = int(data_config['num_of_vertices'])

    miss_type = data_config['miss_type']
    miss_rate = float(data_config['miss_rate'])
    data_prefix = data_config['data_prefix']

    true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")

    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, :1] 

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :1]
    true_data[np.isnan(true_data)] = 0
    mask*= (true_data!=0)

    time = pd.date_range(start=time_config['start'],periods=true_data.shape[0],freq=time_config['freq'])
    dayofweek = np.reshape(time.weekday, (-1, 1))
    print(dayofweek.shape)
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                // (time_slice_size * 60)  # total seconds
    timeofday = np.reshape(timeofday, (-1, 1))
    te = np.concatenate((dayofweek, timeofday), -1)

    # Divide the dataset first ,and construct the sample
    slices = true_data.shape[0]
    train_slices = int(slices * 0.6)
    val_slices = int(slices * 0.2)
    test_slices = slices - train_slices - val_slices

    train_set = true_data[ : train_slices]
    train_mask = mask[ : train_slices]
    train_te = te[ : train_slices]
    train_condamask = get_hist_mask(train_mask)
    train_set[train_mask==0] = 0
    print(train_set.shape)

    val_set = true_data[train_slices : val_slices + train_slices]
    val_mask = mask[train_slices : val_slices + train_slices]
    val_te = te[train_slices : val_slices + train_slices]
    val_condamask = get_hist_mask(val_mask)
    val_set[val_mask==0] = 0
    print(val_set.shape)

    test_set = true_data[-test_slices : ]
    test_te = te[-test_slices : ]
    test_mask = (test_set!=0).astype('int32')
    test_condamask = mask[-test_slices : ]
    print(test_set.shape)

    data_mean, data_std= np.mean(train_set[train_mask==1]), np.std(train_set[train_mask==1])
    


    x_trains, te_trains, cond_trains, ob_trains = SlideWindows(train_set,train_te,train_condamask,train_mask,sample_len)
    x_vals, te_vals, cond_vals, ob_vals = SlideWindows(val_set,val_te,val_condamask,val_mask,sample_len)
    x_tests, te_tests, cond_tests, ob_tests = SlideWindows(test_set,test_te,test_condamask,test_mask,sample_len)

    print("train: ", x_trains.shape)
    print("val: ", x_vals.shape)
    print("test: ", x_tests.shape)
    print("trainTE: ", te_trains.shape)
    print("valTE: ", te_vals.shape)
    print("testTE: ", te_tests.shape)
    print("mean: ", data_mean)
    print("std: ", data_std)

    return x_trains, te_trains, cond_trains, ob_trains,\
            x_vals, te_vals, cond_vals, ob_vals,\
            x_tests, te_tests, cond_tests, ob_tests,\
            data_mean, data_std

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../configurations/PEMSD4_1dim_construct_samples.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    time_slice_size = int(data_config['time_slice_size'])
    train_ratio = float(data_config['train_ratio'])
    val_ratio = float(data_config['val_ratio'])
    test_ratio = float(data_config['test_ratio'])
    num_his = int(training_config['num_his'])
    num_pred = int(training_config['num_pred'])
    num_of_vertices = int(data_config['num_of_vertices'])


    data_file = data_config['data_file']
    files = np.load(data_file, allow_pickle=True)
    data=files['data']
    #timestamp=files['timestamp']
    print(data.shape)
    #print(timestamp)
    print("Dataset: ", data.shape, data[5, 0, :])

    # Divide the dataset first ,and construct the sample
    slices = data.shape[0]
    train_slices = int(slices * 0.6)
    val_slices = int(slices * 0.2)
    test_slices = slices - train_slices - val_slices
    train_set = data[ : train_slices]
    print(train_set.shape)
    val_set = data[train_slices : val_slices + train_slices]
    print(val_set.shape)
    test_set = data[-test_slices : ]
    print(test_set.shape)

    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    xy = {}
    te = {}
    for set_name in sets.keys():
        data_set = sets[set_name]
        X, Y = seq2instance_plus(data_set[..., :1].astype("float64"), num_his, num_pred)

        xy[set_name] = [X, Y]

        time = data_set[:, 0, -1]  # timestamp
        if "PEMSD" in data_file:
            time = pd.to_datetime(time,unit='s')
        time = pd.DatetimeIndex(time)
        dayofweek = np.reshape(time.weekday, (-1, 1))
        print(dayofweek.shape)
        timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                    // (time_slice_size * 60)  # total seconds
        timeofday = np.reshape(timeofday, (-1, 1))
        time = np.concatenate((dayofweek, timeofday), -1)
        time = seq2instance(time, num_his, num_pred)
        te[set_name] = np.concatenate(time, 1).astype(np.int32)

    x_trains, y_trains = xy['train'][0], xy['train'][1]
    x_vals, y_vals = xy['val'][0], xy['val'][1]
    x_tests, y_tests = xy['test'][0], xy['test'][1]

    trainTEs = te['train']
    valTEs = te['val']
    testTEs = te['test']
    print("train: ", x_trains.shape, y_trains.shape)
    print("val: ", x_vals.shape, y_vals.shape)
    print("test: ", x_tests.shape, y_tests.shape)
    print("trainTE: ", trainTEs.shape)
    print("valTE: ", valTEs.shape)
    print("testTE: ", testTEs.shape)
    output_dir = data_config['output_dir']
    output_path = os.path.join(output_dir, "samples_" + str(num_his) + "_" + str(num_pred) + "_" + str(time_slice_size) + ".npz")
    print(f"save file to {output_path}")
    np.savez_compressed(
            output_path,
            train_x=x_trains, train_target=y_trains,
            val_x=x_vals, val_target=y_vals,
            test_x=x_tests, test_target=y_tests,
            trainTE=trainTEs, testTE=testTEs, valTE=valTEs)