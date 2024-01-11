import os
import numpy as np
import argparse
import configparser

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


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


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


def read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename,
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
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
    return all_data



def loadImputationData(config):

    data_config = config['Data']
    training_config = config['Training']

    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    if config.has_option('Data', 'id_filename'):
        id_filename = data_config['id_filename']
    else:
        id_filename = None

    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    dataset_name = data_config['dataset_name']
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])

    miss_type = data_config['miss_type']
    miss_rate = float(data_config['miss_rate'])
    data_prefix = data_config['graph_signal_matrix_filename']

    true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")

    sample_len = num_for_predict

    miss = np.load(miss_datapath)
    mask = miss['mask'][:, :, :1] 

    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :1]
    true_data[np.isnan(true_data)] = 0
    mask*= (true_data!=0)

    te = np.arange(true_data.shape[0]).reshape(-1,1)

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
    print("train mask: ", cond_trains.shape)
    print("val: ", x_vals.shape)
    print("val mask: ", cond_vals.shape)
    print("test: ", x_tests.shape)
    print("test mask: ", cond_tests.shape)
    print("trainTE: ", te_trains.shape)
    print("valTE: ", te_vals.shape)
    print("testTE: ", te_tests.shape)
    print("mean: ", data_mean)
    print("std: ", data_std)

    return x_trains.transpose(0, 2, 3, 1), te_trains, cond_trains.transpose(0, 2, 3, 1), ob_trains.transpose(0, 2, 3, 1),\
            x_vals.transpose(0, 2, 3, 1), te_vals, cond_vals.transpose(0, 2, 3, 1), ob_vals.transpose(0, 2, 3, 1),\
            x_tests.transpose(0, 2, 3, 1), te_tests, cond_tests.transpose(0, 2, 3, 1), ob_tests.transpose(0, 2, 3, 1),\
            data_mean, data_std