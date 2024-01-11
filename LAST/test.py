import pandas as pd
import numpy as np
import os
import argparse
import configparser
from copy import deepcopy
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from data import loadImputationData
from main import Run


config_dir = "./configurations"
miss_types = ["SR-TR","SR-TC","SC-TR","SC-TC"]
miss_rates = [0.1,0.3,0.5,0.7,0.9]


csv_datas = [[],[],[],[],[],[],[],[]]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    for configfile in os.listdir(config_dir):
        config.read(os.path.join(config_dir,configfile))
        data_config = config['Data']
        dataset = data_config['dataset_name']

        for miss_type in miss_types:
            for miss_rate in miss_rates:
                data_config['miss_type'] = miss_type
                data_config['miss_rate'] = str(miss_rate)
        
                [last_result,last_all_result] = Run(config,miss_type,miss_rate)

                csv_datas[0].append("LAST")
                csv_datas[1].append(data_config["dataset_name"])
                csv_datas[2].append(miss_type)
                csv_datas[3].append(miss_rate)
                for i in range(4,8):
                    csv_datas[i].append(last_result[i-4])

                
                csv_datas[0].append("LAST_all")
                csv_datas[1].append(data_config["dataset_name"])
                csv_datas[2].append(miss_type)
                csv_datas[3].append(miss_rate)
                for i in range(4,8):
                    csv_datas[i].append(last_all_result[i-4])


    frame = pd.DataFrame({"model":csv_datas[0],"dataset":csv_datas[1],\
                      "miss_type":csv_datas[2],"miss_rate":csv_datas[3],\
                        "mae":csv_datas[4],"rmse":csv_datas[5],"mape":csv_datas[6],\
                            "time":csv_datas[7]})
    frame.to_csv("result.csv",index=False,sep=',')



