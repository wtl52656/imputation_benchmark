import numpy as np

def Add_Window_Horizon(data,condmask,gtmask, window=3):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, W, ...]
    '''
    length = len(data)
    end_index = length  - window + 1

    data_x = data.copy()
    data_x[condmask==0]=0

    data_y = data.copy()
    data_y[gtmask==0]=0

    X = []      #windows
    Y = []      #horizon
    M = []      #ground truth mask
    index = 0

    while index < end_index:
        X.append(data_x[index:index+window])
        Y.append(data_y[index:index+window])
        M.append(gtmask[index:index+window])
        index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    M = np.array(M)

    return X, Y, M

