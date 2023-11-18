import argparse
import torch
import os
import configparser
from dataset_traffic import get_dataloader
from main_model import CSDI_Traffic
from utils import train, evaluate
import nni

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="config/PEMS04.conf")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)

args = parser.parse_args()
print(args)

config = configparser.ConfigParser()
config.read(args.config)

config["model"]["target_strategy"] = args.targetstrategy
data_prefix = config['file']['data_prefix']

true_datapath = os.path.join(data_prefix,f"true_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
miss_datapath = os.path.join(data_prefix,f"miss_data_{config['train']['type']}_{config['train']['miss_rate']}_v2.npz")
val_ratio = float(config['train']['val_ratio'])
test_ratio = float(config['train']['test_ratio'])
use_nni = int(config['train']['use_nni'])
sample_len = int (config['train']['sample_len'])
batch_size = int (config['train']['batch_size'])
nsample = int(config['diffusion']['nsample']) #diffusion评估时采样次数

if use_nni:
    params = nni.get_next_parameter()

    target_strategy = params['target_strategy']
    config["model"]["target_strategy"] = target_strategy

    timeemb = int(params['timeemb'])
    config['model']['timeemb'] = str(timeemb)

    featureemb = int(params['featureemb'])
    config['model']['featureemb'] = str(featureemb)

    layers = int(params['layers'])
    config['diffusion']['layers'] = str(layers)

    diffusion_embedding_dim = int(params['diffusion_embedding_dim'])
    config['diffusion']['diffusion_embedding_dim'] = str(diffusion_embedding_dim)

    nheads = int(params['nheads'])
    config['diffusion']['nheads'] = str(nheads)

train_loader, valid_loader, test_loader,target_dim,_std,_mean = get_dataloader(
    true_datapath,miss_datapath,val_ratio,test_ratio,batch_size,sample_len
)
model = CSDI_Traffic(config,target_dim).cuda()

if args.modelpath == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader
    )
else:
    model.load_state_dict(torch.load(args.modelpath))

evaluate(
    model,
    test_loader,
    _std,_mean,use_nni,
    nsample=nsample,
)
