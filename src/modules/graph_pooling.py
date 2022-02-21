# -*- coding: utf-8 -*-
"""
Graph pooling
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import GlobalAttention


class GateLayer(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int) -> None:
        super(GateLayer, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh()
        )
        self.h2 = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Sigmoid()
        )
        self.gate_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, seqs):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :return: shape [batch_size, 1]
        """
        gates = self.gate_layer(self.h1(seqs) * self.h2(seqs))
        return gates


class MaskGlobalAttention(GlobalAttention):
    def forward(self, x, batch, mask, size=None):
        """
        x: shape [node_num, in_channel]
        batch: shape [node_num, ]
        mask: shape [node_num, *]

        return: shape [batch_size, *, out_channel]
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        out_list = []
        mask_size = mask.size(1)
        gate = gate.squeeze(1)
        for i in range(mask_size):
            mask_i = mask[:, i]
            gate_mask = gate.masked_fill(mask_i == 0, -1e9)
            gate_mask = softmax(gate_mask.view(-1, 1), batch, num_nodes=size)
            out = scatter_add(gate_mask * x, batch, dim=0, dim_size=size)
            out_list.append(out.unsqueeze(1))

        return torch.cat(out_list, dim=1)
