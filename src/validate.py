# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/12

"""
import os
from tqdm import tqdm

import hydra
import torch
from torch_geometric.data import DataLoader

from datasets.graph_datasets import ValidationDataset
from models.kg_model import Model
from utils.train_util import set_seed
from utils.eval_util import group_labels
from utils.eval_util import cal_metric


def validate(cfg, device):
    set_seed(cfg.training.seed)

    model = Model(cfg)
    saved_model_path = os.path.join(cfg.training.model_save_path, 'model.ep{0}'.format(cfg.training.validate_epoch))
    print("Load from:", saved_model_path)
    if not os.path.exists(saved_model_path):
        print("Not Exist: {}".format(saved_model_path))
        return []
    model.cpu()
    pretrained_model = torch.load(saved_model_path, map_location='cpu')
    model.load_state_dict(pretrained_model, strict=False)
    model.title_embedding = model.title_embedding.to(device)
    model.to(device)
    model.eval()

    epoch = cfg.training.validate_epoch
    preds, truths, imp_ids = list(), list(), list()
    for split in range(10):
        valid_dataset = ValidationDataset(cfg.dataset.root_path, split, "valid_set_dist", "valid_set.pt")
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=cfg.training.batch_size, shuffle=False)

        # Setting the tqdm progress bar
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test: {} on split {}".format(cfg.training.validate_epoch, split),
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")

        with torch.no_grad():
            for i, data in data_iter:
                imp_ids += data.imp_id.cpu().numpy().tolist()
                data = data.to(device)

                # 1. Forward
                pred = model.validation_step(data)

                preds += pred.cpu().numpy().tolist()
                truths += data.y.long().cpu().numpy().tolist()

    all_labels, all_preds = group_labels(truths, preds, imp_ids)
    metric_list = [x.strip() for x in cfg.training.metrics.split("||")]
    ret = cal_metric(all_labels, all_preds, metric_list)
    for metric, val in ret.items():
        print("Epoch: {}, {}: {}".format(epoch, metric, val))


@hydra.main(config_path="../conf/train.yaml")
def main(cfg):
    set_seed(cfg.training.seed)

    if cfg.training.gpus == 0:
        print("== CPU Mode ==")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    validate(cfg, device)


if __name__ == '__main__':
    main()
