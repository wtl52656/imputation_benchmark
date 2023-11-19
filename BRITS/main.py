import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import pickle
import numpy as np
import time
import utils
import models
import argparse
import data_loader
from tqdm import tqdm
import metrics
import configparser
import os
from copy import deepcopy

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from metrics import masked_mape_np

parser = argparse.ArgumentParser()

parser.add_argument("--for_test", type=int, default=0) # 调试模式：跑部分数据
parser.add_argument("--config", default='configurations/PEMS04_12_SR-TC_0.5.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()


config = configparser.ConfigParser()
config.read(args.config)
config = config["train"]
epochs = int(config['epochs'])
nodes = int(config['nodes'])
seq_len = int(config['seq_len'])
attributes = int(config['attributes'])
learning_rate = float(config['learning_rate'])
hid_size = int(config['hid_size'])
batch_size = int(config['batch_size'])
patience = int(config['patience'])
file_prefix = config["file_prefix"]

f= open(os.path.join(file_prefix,f"data_meanstd_{config['type']}_{config['miss_rate']}.pkl"),"rb")
mead_std = pickle.load(f)

mean = mead_std['mean']
std = mead_std['std']

use_nni = int(config['use_nni'])

if use_nni:
    import nni
    params = nni.get_next_parameter()
    hid_size = int(params['hid_size'])
    config['hid_size'] = str(hid_size)
    learning_rate = float(params['learning_rate'])
    config['learning_rate'] = str(learning_rate)


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_path = os.path.join(file_prefix,f"data_train_{config['type']}_{config['miss_rate']}.json")
    print(train_path)
    test_path = os.path.join(file_prefix,f"data_test_{config['type']}_{config['miss_rate']}.json")
    val_path = os.path.join(file_prefix,f"data_val_{config['type']}_{config['miss_rate']}.json")

    train_iter = data_loader.get_loader(train_path, batch_size=batch_size) # 训练集打乱数据顺序！
    test_iter = data_loader.get_loader(test_path, batch_size=batch_size, shuffle=False) # 测试集不打乱数据顺序！
    val_iter = data_loader.get_loader(val_path, batch_size=batch_size, shuffle=False)

    print (f'train_set_size:{len(train_iter)}\ntest_set_size:{len(test_iter)}\nval_set_size:{len(val_iter)}\n')

    for data in train_iter:
        print("train sample: ", data['forward']['values'].shape, flush=True)
        break
    for data in test_iter:
        print("test sample: ", data['forward']['values'].shape, flush=True)
        break
    for data in val_iter:
        print("val sample: ", data['forward']['values'].shape, flush=True)
        break

    model_path = os.path.join(config['experiment_path'],f"brits_{config['type']}_{config['miss_rate']}_{hid_size}_{batch_size}.params")

    impatience = 0
    best_loss = 1e9
    best_epoch = -1
    best_model = deepcopy(model.state_dict())

    train_all_time = 0

    epoch_counter = 0

    start_train = time.time()

    for epoch in range(epochs):
        epoch_counter += 1
        start = time.time()

        model.train()

        run_loss = 0.0

        for idx, data in enumerate(train_iter):
            if args.for_test and idx == 5:
                break

            # print(f"{idx}: ", data)
            data = utils.to_var(data)   #转成Variable类型并放到显存中
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            #print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        end = time.time()

        train_all_time += end - start

        print(f"epoch {epoch} train spend {round(end - start,2)} seconds", flush=True)

        start = time.time()

        val_loss = evaluate(model, val_iter)

        print(f"epoch {epoch} val spend {round(time.time() - start,2)} seconds", flush=True)

        # 早停机制
        if val_loss < best_loss:
            print(f"Validation loss decrease from {best_loss} ot {val_loss}")
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch
            impatience = 0
        else:
            impatience = impatience + 1
            print(f"Patience: {impatience}/{patience}")
        if impatience >= patience:
            print('Breaking due to early stopping at epoch %d， best epoch at %d' % (epoch, best_epoch), flush=True)
            break


        if args.for_test:
            break # 小数据集测试模式下，只跑一轮就结束
    
    if use_nni:
        nni.report_final_result(best_loss)

    end_train = time.time()

    torch.save(best_model, model_path)

    # 加载最佳模型
    model.load_state_dict(best_model)
    # print(model_path)
    # model.load_state_dict(torch.load(model_path))


    start = time.time()

    mae = evaluate(model, test_iter,val_set=False, save=True)

    end=time.time()

    print(f"test spend {round(end - start,2)} seconds", flush=True)
    print(f"all epochs spend {round(train_all_time,2)} seconds,\t average epoch spend {round(train_all_time/epoch_counter,2)} seconds")
    print(f"train and val spend {round(end_train - start_train,2)} seconds")


def evaluate(model, val_iter, val_set=True, save=False):
    model.eval()

    with torch.no_grad():

        evals = []
        imputations = []

        if val_set:
            print("Validation...")

            run_loss = 0

            for idx, data in enumerate(val_iter):
                if args.for_test and idx == 5:
                    break

                data = utils.to_var(data) 
                ret = model.run_on_batch(data, None)

                run_loss += ret['loss'].item()
            
            return run_loss

        else:
            print("Test...")

            impute = []
            for idx, data in enumerate(val_iter):
                if args.for_test and idx == 5:
                    break

                data = utils.to_var(data)
                ret = model.run_on_batch(data, None)
                #==============
                impute.append(ret['imputations'].data.cpu().numpy())
                #==============
                imputation = ret['imputations'].data.cpu().numpy()  #插补值
                true_data = data['forward']['true_data'].cpu().numpy()#真实值
                mask = data['forward']['masks'].cpu().numpy() 
                evals.append(true_data[np.where(mask == 0)])
                imputations.append(imputation[np.where(mask == 0)]) #仅拿缺失处计算指标

            evals = np.concatenate(evals, axis=0)
            imputations = np.concatenate(imputations, axis=0)


            imputations = imputations * std + mean#反归一化
            #================================
            impute = np.concatenate(impute, axis=0)#BT(N*F)
            # print(type(impute))
            impute = torch.tensor(impute)
            # print(type(impute), impute.shape)
            impute = impute*std + mean
            # print(impute.shape, type(impute))
            impute_saved = impute[0]#T(N*F)
            for i in range(1,impute.shape[0]):
                impute_saved = torch.cat((impute_saved,impute[i]),0)
            impute_saved = impute_saved.unsqueeze(dim=2)#T*N*1
            # print(impute_saved.shape)
            # np.savez_compressed('BRITS_SR-TC_0.5_1009_reshape.npz',impute = impute_saved.cpu().numpy())

            #================================
            print("test evals: ", evals.shape, evals[:25])
            print("test imputations: ", imputations.shape, imputations[:25])

            mae = (np.abs(evals[...] - imputations[...]) ).mean()
            print("Test MAE: ", mae)

            # 评测第一个维度
            # mae_new = np.sum(np.abs(evals - imputations)) / evals.shape[0]              # LATC

            mae = mean_absolute_error(evals[...], imputations[...])
            # print("MAE compare | ori: %.4f, new1: %.4f" %(mae, mae_new))
            rmse = mean_squared_error(evals[...], imputations[...])**0.5                     #,squared=False
            # rmse_new = np.sqrt(np.sum((evals - imputations) ** 2) / evals.shape[0])     # LATC
            # print("RMSE compare | ori: %.4f, new1: %.4f" %(rmse, rmse_new))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.nan_to_num(np.abs((evals - imputations) / evals),nan=0, posinf=0, neginf=0)) * 100
                # mape_new = np.sum(np.abs(evals - imputations) / evals) / evals.shape[0]     # LATC
            # print("MAPE compare | ori: %.4f, new1: %.4f" %(mape, mape_new))
            print('Test:\t MAE:%.4f\t RMSE:%.4f\t MAPE:%.4f\t' % (mae, rmse, mape))
            

        return mae



def run():
    model = getattr(models,config['model']).Model(hid_size, attributes*nodes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    ctx = config['cuda']
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()
