import numpy as np


def masked_mape(y_true, y_pred, mask):
    mask = mask.astype('long')
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs(y_pred - y_true) / y_true
        mask = mask & ~np.isnan(mape) & ~np.isinf(mape)
        mape = np.nan_to_num(mape, posinf=0)
        mape = mask * mape
        return np.sum(mape) / np.sum(mask)


def masked_mae(y_true, y_pred, mask):
    mae = np.abs(y_true - y_pred)
    mae = mae * mask
    return np.sum(mae) / np.sum(mask)


def masked_rmse(y_true, y_pred, mask):
    return np.sqrt(masked_mse(y_true, y_pred, mask))


def masked_mse(y_true, y_pred, mask):
    mse = (y_true - y_pred) ** 2
    mse = mse * mask
    return np.sum(mse) / np.sum(mask)