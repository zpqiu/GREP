# -*- encoding:utf-8 -*-
"""
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
import os
import argparse
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from datasets.graph_datasets import NewsDataset
from datasets.graph_datasets import ValidationDataset
from models.kg_model import Model
from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric


def run(cfg: DictConfig, rank: int, device: torch.device, train_dataset: NewsDataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    set_seed(cfg.training.seed)
    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)

    # # Build model.
    model = Model(cfg)
    # model = build_model(cfg)
    #
    # Build optimizer.
    steps_one_epoch = len(train_data_loader) // cfg.training.accumulate
    train_steps = cfg.training.epochs * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-5)

    model.title_embedding = model.title_embedding.to(device)
    model.to(device)
    
    
    print("Worker %d is working ... " % rank)
    # Fast check the validation process
    if (cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0):
        validate(cfg, -1, model, device, fast_dev=True)
    
    # Training and validation
    for epoch in range(cfg.training.epochs):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, rank, model, train_data_loader,
              optimizer, steps_one_epoch, device)
    
        if (cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0):
            validate(cfg, epoch, model, device)
            save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.training.model_save_path)


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def train(cfg, epoch, rank, model, loader, optimizer, steps_one_epoch, device):
    """
    train loop
    :param args: config
    :param epoch: int, the epoch number
    :param gpu_id: int, the gpu id
    :param rank: int, the process rank, equal to gpu_id in this code.
    :param model: gating_model.Model
    :param loader: train data loader.
    :param criterion: loss function
    :param optimizer:
    :param steps_one_epoch: the number of iterations in one epoch
    :return:
    """
    model.train()

    model.zero_grad()

    enum_dataloader = enumerate(loader)
    if ((cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0)):
        enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch * cfg.training.accumulate:
            break
        # data = {key: value.to(device) for key, value in data.items()}
        data = data.to(device)
        # 1. Forward
        loss = model.training_step(data)

        if cfg.training.accumulate > 1:
            loss = loss / cfg.training.accumulate

        # 3.Backward.
        loss.backward()

        if ((cfg.training.gpus < 2) or (cfg.training.gpus > 1 and rank == 0)) and ((i+1) % cfg.logger.log_freq == 0):
            # neptune.log_metric("loss", loss.item())
            print("loss", loss.item())

        if (i + 1) % cfg.training.accumulate == 0:
            if cfg.training.gpus > 1:
                average_gradients(model)
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()

    # if (not args.dist_train) or (args.dist_train and rank == 0):
    #     util.save_checkpoint_by_epoch(
    #         model.state_dict(), epoch, args.checkpoint_path)


def validate(cfg, epoch, model, device, fast_dev=False):
    model.eval()

    valid_dataset = NewsDataset(cfg.dataset.root_path, cfg.dataset.valid_split_cnt, "valid_set_dist", "valid_set_sample.pt")
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Setting the tqdm progress bar
    data_iter = tqdm(enumerate(valid_data_loader),
                     desc="EP_test:%d" % epoch,
                     total=len(valid_data_loader),
                     bar_format="{l_bar}{r_bar}")

    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:
            if fast_dev and i > 10:
                break

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


def init_processes(cfg, local_rank, vocab, dataset, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = cfg.training.master_port
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.training.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, train_dataset=dataset)


def set_log_service(api_token, params, project_name="mind", exp_name="base_gat"):
    train_dir = os.path.dirname(__file__)
    # neptune.init(project_name, api_token=api_token)
    # neptune.create_experiment(name=exp_name, params=params, upload_source_files=[train_dir + '/models/kg_model.py'])


def split_dataset(dataset, gpu_count):
    sub_len = len(dataset) // gpu_count
    if len(dataset) != sub_len * gpu_count:
        len_a, len_b = sub_len * gpu_count, len(dataset) - sub_len * gpu_count
        dataset, _ = torch.utils.data.random_split(dataset, [len_a, len_b])

    return torch.utils.data.random_split(dataset, [sub_len, ] * gpu_count)


def init_exp(cfg):
    if not os.path.exists(cfg.training.model_save_path):
        os.mkdir(cfg.training.model_save_path)


@hydra.main(config_path="../conf/train.yaml")
def main(cfg):
    # init_exp(cfg)
    set_seed(cfg.training.seed)
    # tokenizer = RobertaTokenizer.from_pretrained(cfg.model.encoder)
    # vocab = WordVocab.load_vocab(cfg.dataset.vocab)
    datasets = NewsDataset(cfg.dataset.root_path, cfg.dataset.train_split_cnt, "training_set_dist", "training_set.pt")

    if cfg.training.gpus == 0:
        print("== CPU Mode ==")
        datasets = NewsDataset(cfg.dataset, cfg.dataset.train)
        run(cfg, 0, datasets, torch.device("cpu"))
    elif cfg.training.gpus == 1:
        datasets = NewsDataset(cfg.dataset, 2, "training_set_dist", "training_set.pt")
        run(cfg, 0, datasets, torch.device("cuda:0"))
    else:
        dataset_list = split_dataset(datasets, cfg.training.gpus)

        processes = []
        for rank in range(cfg.training.gpus):
            p = mp.Process(target=init_processes, args=(
                cfg, rank, None, dataset_list[rank], run, "nccl"))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
