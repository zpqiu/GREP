# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/10

Some helper functions for building model
"""
import torch
import torch.nn as nn
import numpy as np


def build_embedding_layer(pretrained_embedding_path, vocab, embedding_dim):
    num_embeddings = len(vocab)
    if pretrained_embedding_path != "":
        weights = np.load(pretrained_embedding_path)
        weights = torch.tensor(weights).float()
        assert list(weights.size()) == [num_embeddings, embedding_dim]
        print("load pre-trained embeddings.")
        return nn.Embedding.from_pretrained(weights, freeze=False)

    return nn.Embedding(num_embeddings, embedding_dim)
