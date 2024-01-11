import os
import numpy as np
import argparse
import configparser

import torch


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, miss_data_sequence,mask_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    #print("label_start_idx", label_start_idx)
    #print("num_for_predict", num_for_predict)
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(miss_data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([miss_data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(miss_data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([miss_data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(miss_data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([miss_data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    mask = np.concatenate([mask_sequence[i: j]
                             for i, j in hour_indices], axis=0)
    '''
    print(np.array(hour_sample).shape)
    print(np.array(target).shape)
    print(np.array(mask).shape)
    print("^^^^^^^^^^^^^^^")
    '''

    return week_sample, day_sample, hour_sample, target,mask
def MinMaxnormalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def cal_timedelta(mask):
    mask = torch.from_numpy(mask).type(torch.FloatTensor)
    delta = torch.zeros_like(mask)
    print(mask.size())
    num_of_sample,num_of_node,seq_len = mask.shape
    # for i in range(num_of_sample):
    #     for j in range(num_of_node):
    #         ancher = 0
    #         for g in range(seq_len):
    #             if (mask[i,j,g]==0):
    #                 ancher += 1
    #             else:
    #                 ancher = 0
    #             delta[i, j, g] = ancher
    ancher = torch.zeros_like(mask[...,0])
    for g in range(seq_len):
        ancher += (mask[...,g]==0)
        delta[...,g] = ancher
    return delta

def get_timestamp(timestamp):
    timestamp = torch.from_numpy(timestamp).to(torch.float32)
    new_time = timestamp
    for i in range(12-1):
        new_time = torch.cat([timestamp - i-1,new_time],dim=-1)
    new_time = (new_time - new_time[:,0].unsqueeze(-1)) / 11.0

    return new_time


def read_and_generate_dataset_encoder_decoder(all_datapath,true_datapath,miss_datapath,
                                              num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict,
                                              points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(true_datapath)['data'].astype(np.float32)[...,:1]# (sequence_length, num_of_vertices, num_of_features)
    mask_seq = np.load(miss_datapath)['mask'][...,:1]

    print(data_seq.shape)
    print(mask_seq.shape)

    miss_data_seq = np.load(miss_datapath)['data'].astype(np.float32)[...,:1]

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq,miss_data_seq,mask_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        #print(np.array(sample[-2]).shape)
        #print(np.array(sample[-1]).shape)
        #print("^^^^^^^^^^^^^^^^")
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target,mask = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            #print(hour_sample.shape)
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            #print(hour_sample.shape)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        #print(target.shape)
        sample.append(target)
        mask = np.expand_dims(mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        #print(mask.shape)
        sample.append(mask)
        #print("%%%%%%%%%%%%%%%%%%%%%%")

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,mask,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    test_samples = []
    data_seq = data_seq[int(len(data_seq)*0.8):]
    num_of_series = int (data_seq.shape[0] / 12)
    for i in range(num_of_series):
        test_idx = (i) * 12

        sample = get_sample_indices(data_seq,miss_data_seq,mask_seq, num_of_weeks, num_of_days,
                                    num_of_hours, test_idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue
        week_sample, day_sample, hour_sample, target,mask = sample
        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)
        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)
        if num_of_hours > 0:
            #print(hour_sample.shape)
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            #print(hour_sample.shape)
            sample.append(hour_sample)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        #print(target.shape)
        sample.append(target)
        mask = np.expand_dims(mask, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(mask)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)
        test_samples.append(sample)


    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line2])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1:split_line2:])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*test_samples)]

    train_x = np.concatenate(training_set[:-3], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-3], axis=-1)
    test_x = np.concatenate(testing_set[:-3], axis=-1)

    train_target = training_set[-3]  # (B,N,T)
    ######## new train_target
    #train_target = np.concatenate((train_target,train_target),axis=)
    #######################
    val_target = validation_set[-3]
    test_target = testing_set[-3]

    train_mask = training_set[-2]  # (B,N,T)
    val_mask = validation_set[-2]
    test_mask = testing_set[-2]

    ############### process train set
    num_of_samples, num_of_vertices, num_of_features, seq_len = train_x.shape
    train_mask_all = np.expand_dims(train_mask, 2).repeat(num_of_features, 2)
    trainprob_matrix = np.random.rand(num_of_samples, num_of_vertices, num_of_features, seq_len)
    prob_matrix = np.where(trainprob_matrix > 0.5, 1, 0)
    prob_matrix_1 = np.where(train_mask_all == 0, 0, prob_matrix)
    train_x_1 = np.where(prob_matrix_1 == 0, 0, train_x)
    prob_matrix_2 = np.where(train_mask_all == 0, 0, 1 - prob_matrix)
    train_x_2 = np.where(prob_matrix_2 == 0, 0, train_x)
    train_x = np.concatenate((train_x_1,train_x_2),axis=0)
    #train_x = np.concatenate((train_x_1, train_x_2), axis=-2)
    #train_mask = np.concatenate((train_mask,train_mask),axis=0)
    #train_prob_matrix = np.concatenate((prob_matrix_1,prob_matrix_2),axis=0)

    #################################



    ############### process val set
    num_of_samples, num_of_vertices, num_of_features,seq_len = val_x.shape
    val_mask_all = np.expand_dims(val_mask, 2).repeat(num_of_features, 2)
    prob_matrix = np.random.rand(num_of_samples, num_of_vertices, num_of_features,seq_len)
    prob_matrix = np.where(prob_matrix > 0.1, 1, 0)
    prob_matrix = np.where(val_mask_all==0,0,prob_matrix)
    val_x = np.where(prob_matrix==0,0,val_x)
    ##########################################


    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    train_timestamp = get_timestamp(train_timestamp)
    #train_timestamp = np.concatenate((train_timestamp,train_timestamp),axis=0)
    val_timestamp = get_timestamp(val_timestamp)
    test_timestamp = get_timestamp(test_timestamp)
    print(type(val_timestamp))
    '''
    train_coeffs = get_coeffs(train_timestamp,train_x,train_mask)
    val_coeffs = get_coeffs(val_timestamp, val_x, val_mask)
    test_coeffs = get_coeffs(test_timestamp, test_x, test_mask)
    '''
    #train_delta = cal_timedelta(train_mask)
    train_delta1 = cal_timedelta(prob_matrix_1[:,:,0,:])
    train_delta2 = cal_timedelta(prob_matrix_2[:, :, 0, :])
    #val_delta = cal_timedelta(val_mask)
    val_delta = cal_timedelta(prob_matrix[:,:,0,:])
    test_delta = cal_timedelta(test_mask)



    # max-min normalization on x
    (stats, train_x_norm, train_x1_norm, train_x2_norm) = MinMaxnormalization(train_x, train_x_1, train_x_2)
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x1': train_x1_norm,
            'x2': train_x2_norm,
            'target': train_target,
            'mask': train_mask,
            'timestamp': train_timestamp,
            'delta1': train_delta1,
            'delta2': train_delta2,
           # 'coeffs': train_coeffs,

        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'mask': val_mask,
            'timestamp': val_timestamp,
            'delta': val_delta,
         #   'coeffs': val_coeffs,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'mask': test_mask,
            'timestamp': test_timestamp,
            'delta': test_delta,
         #   'coeffs': test_coeffs,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
        }
    }
    print('train x:', all_data['train']['x1'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train mask:', all_data['train']['mask'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val mask:', all_data['val']['mask'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test mask:', all_data['test']['mask'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])

    if save:
        print('save file:', all_datapath)
        np.savez_compressed(all_datapath,
                            train_x1=all_data['train']['x1'],train_x2=all_data['train']['x2'], train_target=all_data['train']['target'],train_mask=all_data['train']['mask'],
                            train_timestamp=all_data['train']['timestamp'],
                            train_delta1=all_data['train']['delta1'],train_delta2=all_data['train']['delta2'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],val_mask=all_data['val']['mask'],
                            val_timestamp=all_data['val']['timestamp'],
                            val_delta=all_data['val']['delta'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],test_mask=all_data['test']['mask'],
                            test_timestamp=all_data['test']['timestamp'],
                            test_delta=all_data['test']['delta'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
    return all_data