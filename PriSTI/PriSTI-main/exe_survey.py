import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np
import nni
from dataset_survey import get_dataloader
from main_model import PriSTI_survey
from utils import train, evaluate
import nni

def load_nni_parameter(config):
    params = nni.get_next_parameter()
    for key,value in params.items():
        for section in config.keys():
            if key in config[section].keys():
                config[section][key] = value

def main(args):
    SEED = args.seed
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    path = args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config['train']['nni'] = args.use_nni
    if config['train']['nni']:
        load_nni_parameter(config)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["seed"] = ""
    print(json.dumps(config, indent=4))

    data_prefix = config['file']['data_prefix']
    miss_type = config['file']['miss_type']
    miss_rate = float(config['file']['miss_rate'])
    dataset = config['file']['dataset']

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/" + dataset+ '_'+ miss_type +'_'+str(miss_rate) + '_' + current_time + "/"
    )

    train_loader, valid_loader, test_loader, scaler, mean_scaler,node_num = get_dataloader(
        config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy,data_prefix=data_prefix,miss_type=miss_type,miss_rate=miss_rate
    )
    config["diffusion"]["node_num"] = node_num

    model = PriSTI_survey(config, args.device,target_dim = node_num).to(args.device)

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    test_mae,_,_,_ = evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
    if config['train']['nni']:
        nni.report_final_result(test_mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="./config/pems04.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="block")     # block|point
    parser.add_argument("--use_nni", action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)
