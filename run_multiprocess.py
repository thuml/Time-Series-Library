import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.print_args import print_args


def myprint(*args):
    sys.stdout.write(' '.join(map(str, args)) + '\n')
    logging.basicConfig(
        format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(process)s-%(levelname)s: %(message)s',
        level=logging.INFO,
        filename='./log/out.log',
        filemode='a')
    logging.info(' '.join(map(str, args)))


def main(path):
    # builtins.print = myprint
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    with open(path, 'r') as f:
        args = f.read()
    args = argparse.Namespace(**json.loads(args))
    if not args.is_training:
        return
    args.num_workers = 4

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
        args.is_training = 0
        with open(path, 'w') as f:
            json.dump(args.__dict__, f)
            f.close()
        logging.debug("=" * 10 + f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} COMPLETE" + "=" * 10)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


def get_json_path(dir_path):
    def fun(dir_path, list_name):
        """递归获取目录下（文件夹下）所有文件的路径"""
        for file in os.listdir(dir_path):  # 获取文件（夹）名
            file_path = os.path.join(dir_path, file)  # 将文件（夹）名补全为路径
            if os.path.isdir(file_path):  # 如果是文件夹，则递归
                fun(file_path, list_name)
            else:
                if file_path.endswith('.json'):
                    list_name.append(file_path)  # 保存路径
        return list_name

    res = []
    fun(dir_path, res)
    return res


if __name__ == '__main__':
    with open(r'./log/out.log', 'a+', encoding='utf-8') as f:
        f.truncate(0)
    #  builtins.print = myprint
    json_path = (get_json_path("./scripts/short_term_forecast")
                 + get_json_path("./scripts/long_term_forecast/ETT_script")
                 + get_json_path("./scripts/long_term_forecast/Exchange_script")
                 + get_json_path("./scripts/long_term_forecast/ILI_script"))
    json_path.sort()
    multi_process = 0
    if multi_process > 1:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=multi_process)
        task = []
        for path in json_path:
            pool.submit(main, path)
        # 等待任务执行完, 也可以设置一个timeout时间
        # wait(task, return_when=ALL_COMPLETED)
        pool.shutdown(wait=True)
        #
        print('main process done')
    else:
        for path in json_path:
            main(path)
