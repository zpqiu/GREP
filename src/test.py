# -*- encoding:utf-8 -*-
"""
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
from torch_geometric.data import DataLoader

from datasets.graph_datasets import NewsDataset
from datasets.graph_datasets import ValidationDataset
from models.kg_model import Model
from utils.train_util import set_seed


def run(cfg: DictConfig, rank: int, device: torch.device):
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

    st = rank * 10
    ed = st + 10
    for split in range(st, ed):
        test_dataset = ValidationDataset(cfg.dataset.root_path, split, "{}/test_set_dist".format(cfg.dataset.test_split_name), "test_set.pt")
        valid_data_loader = DataLoader(
            test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

        if ((cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0)):
            data_iter = tqdm(enumerate(valid_data_loader),
                             desc="EP_test:%d" % 1,
                             total=len(valid_data_loader),
                             bar_format="{l_bar}{r_bar}")
        else:
            data_iter = enumerate(valid_data_loader)

        with torch.no_grad():
            preds, truths, imp_ids = list(), list(), list()
            for i, data in data_iter:
                imp_ids += data.imp_id.cpu().numpy().tolist()
                data = data.to(device)

                # 1. Forward
                pred = model.validation_step(data)

                preds += pred.cpu().numpy().tolist()
                truths += data.y.long().cpu().numpy().tolist()

            result_file_path = os.path.join(cfg.dataset.result_path, "{}_{}.txt".format(cfg.dataset.test_split_name, split))

            with open(result_file_path, 'w', encoding='utf-8') as f:
                for imp_id, truth, pred in zip(imp_ids, truths, preds):
                    f.write("{}\t{}\t{}\n".format(imp_id, truth, pred))


@hydra.main(config_path="../conf/train.yaml")
def main(cfg):
    # init_exp(cfg)
    set_seed(cfg.training.seed)

    if cfg.training.gpus == 0:
        print("== CPU Mode ==")
        datasets = NewsDataset(cfg.dataset, cfg.dataset.train)
        run(cfg, 0, datasets, torch.device("cpu"), datasets)
    elif cfg.training.gpus == 1:
        # datasets = NewsDataset(cfg.dataset, 2, "training_set_dist", "training_set.pt")
        run(cfg, 0, torch.device("cuda:0"))
    else:
        processes = []
        for rank in range(cfg.training.gpus):
            cur_device = torch.device("cuda:{}".format(rank))

            p = mp.Process(target=run, args=(cfg, rank, cur_device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
