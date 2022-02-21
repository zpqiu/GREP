# -*- encoding:utf-8 -*-
"""
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
import argparse
from tqdm import tqdm
import json
import scipy.stats as ss
import numpy as np
import pandas as pd
import hydra
import math
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from datasets.graph_datasets import NewsDataset
from datasets.graph_datasets import ValidationDataset, DistValidationDataset
from models.kg_model import Model
from utils.eval_util import group_labels
from utils.eval_util import cal_metric



def gather(cfg, filenum, validate=False):
    output_path = cfg.dataset.result_path

    preds = []
    labels = []
    imp_indexes = []

    for i in range(filenum):
        with open(output_path + 'tmp_{}.json'.format(i), 'r', encoding='utf-8') as f:
            cur_result = json.load(f)
        imp_indexes += cur_result['imp']
        labels += cur_result['labels']
        preds += cur_result['preds']

    print(len(preds))
    all_keys = list(set(imp_indexes))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, imp_indexes):
        group_labels[k].append(l)
        group_preds[k].append(p)
    
    if validate:
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        
        metric_list = [x.strip() for x in cfg.training.metrics.split("||")]
        ret = cal_metric(all_labels, all_preds, metric_list)
        for metric, val in ret.items():
            print("Epoch: {}, {}: {}".format(1, metric, val))

    final_arr = []
    for k in group_preds.keys():
        new_row = []
        new_row.append(k)
        new_row.append(','.join(list(map(str, np.array(group_labels[k]).astype(int)))))
        new_row.append(','.join(list(map(str, np.array(group_preds[k]).astype(float)))))
        
        rank = ss.rankdata(-np.array(group_preds[k])).astype(int).tolist()
        new_row.append('[' + ','.join(list(map(str, rank))) + ']')
        
        assert(len(rank) == len(group_labels[k]))
        
        final_arr.append(new_row)
    
    fdf = pd.DataFrame(final_arr, columns=['impression', 'labels', 'preds', 'ranks'])
    fdf.drop(columns=['labels', 'ranks']).to_csv(output_path + 'score.txt', sep=' ', index=False)
    fdf.drop(columns=['labels', 'preds']).to_csv(output_path + 'result.txt', header=None, sep=' ', index=False)

@hydra.main(config_path="../conf/train.yaml")
def main(cfg):
    
    gather(cfg, 20)
        



if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
