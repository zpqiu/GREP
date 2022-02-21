# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/01

Dataset class
"""
import os
import pickle
import torch
import json
from itertools import repeat

from tqdm import tqdm
from omegaconf import DictConfig

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_geometric
import math


class KGData(Data):
    def __init__(self, ents=None, news=None, edge_index=None, bi_edge_index=None, y=None):
        # super(KGData, self).__init__(x=ents, edge_index=edge_index, y=y)
        self.x = ents
        self.edge_index = edge_index
        self.edge_attr = None
        self.y = y
        self.pos = None
        self.normal = None
        self.face = None
        if torch_geometric.is_debug_enabled():
            self.debug()

        self.bi_edge_index = bi_edge_index
        self.news = news

    def __inc__(self, key, value):
        if key == 'bi_edge_index':
            return torch.tensor([[self.news.size(0)], [self.x.size(0)]])
        else:
            return super(KGData, self).__inc__(key, value)
    
    def _long(self):
        self.bi_edge_index = self.bi_edge_index.long()
        self.edge_index = self.edge_index.long()


class NewsDataset(InMemoryDataset):
    def __init__(self, root_path: str, splits: int, raw_base_filename: str, processed_file_name: str, transform=None, pre_transform=None):
        self.root_path = root_path
        self.splits = splits
        self.raw_base_filename = raw_base_filename
        self.processed_file_name = processed_file_name

        super(NewsDataset, self).__init__(self.root_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data._long()

    @property
    def raw_dir(self):
        return self.root_path

    @property
    def processed_dir(self):
        return os.path.join(self.root_path, 'processed')

    @property
    def raw_file_names(self):
        return ["{}-{}.pt".format(self.raw_base_filename, i) for i in range(self.splits)]

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def convert_to_list(self, in_data, in_slices):

        def _convert_one(idx):
            data = in_data.__class__()
            if hasattr(in_data, '__num_nodes__'):
                data.num_nodes = in_data.__num_nodes__[idx]

            for key in in_data.keys:
                item, slices = in_data[key], in_slices[key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[in_data.__cat_dim__(key, item)] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]
            return data
        def _len():
            for item in in_slices.values():
                return len(item) - 1
            return 0
        cnt = _len()
        return [_convert_one(i) for i in range(cnt)]

    def process(self):
        data_list = []

        for file_idx, raw_path in enumerate(self.raw_paths):
            print("Processing-{}".format(raw_path))
            data, slices = torch.load(raw_path)
            data_list += self.convert_to_list(data, slices)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ValidationDataset(InMemoryDataset):
    def __init__(self, root_path: str, split: int, raw_base_filename: str, processed_file_name: str, transform=None, pre_transform=None):
        self.root_path = root_path
        self.splits = split
        self.raw_base_filename = raw_base_filename
        self.processed_file_name = processed_file_name

        super(ValidationDataset, self).__init__(self.root_path, transform, pre_transform)
        source_file_path = os.path.join(self.root_path, "{}-{}.pt".format(self.raw_base_filename, split))
        print("Loading", source_file_path)
        self.data, self.slices = torch.load(source_file_path)
        self.data._long()
