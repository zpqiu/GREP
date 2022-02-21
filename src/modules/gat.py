# -*- coding: utf-8 -*-
"""GAT module

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from modules.transformer_conv import TransformerConv
from modules.cross_conv import CrossConv


class GatNet(nn.Module):
    def __init__(self, node_feat_size, conv_hidden_size, dropout=0.2):
        super(GatNet, self).__init__()
        self.conv1 = TransformerConv(node_feat_size, conv_hidden_size // 2, dropout=dropout, heads=2, beta=0.0)

        self.gate_layer = nn.Sequential(
            nn.Linear(2 * node_feat_size, node_feat_size, bias=False),
            nn.Sigmoid()
        )
        self.fuse_layer = nn.Sequential(
            nn.Linear(node_feat_size, node_feat_size, bias=False),
            nn.Tanh()
        )
        self.output_proj = nn.Linear(node_feat_size, node_feat_size)
        # self.layer_norm = nn.LayerNorm(node_feat_size)

    def forward(self, nodes, edge_index):
        """
        nodes: shape [*, node_feat_size]
        edge_index: shape [2, *]
        """
        x = self.conv1(nodes, edge_index)
        # x = self.output_proj(x)
        h = torch.cat([nodes, x], dim=-1)
        gate = self.gate_layer(h)
        output = gate * self.fuse_layer(x) + (1.0 - gate) * nodes
        # output = self.gate_layer(h) * self.fuse_layer(h)
        
        # output = self.layer_norm(x + nodes)
        return output


class CrossGatNet(nn.Module):
    def __init__(self, feat_size, conv_hidden_size, flow, dropout=0.2):
        super(CrossGatNet, self).__init__()
        self.conv1 = CrossConv((feat_size, feat_size), conv_hidden_size // 2, dropout=dropout, heads=2, flow=flow)
        # self.conv1 = GATConv((feat_size, feat_size), conv_hidden_size // 2, dropout=dropout, heads=2, flow=flow, add_self_loops=False)
        # self.conv1 = TransformerConv((feat_size, feat_size), conv_hidden_size // 2, dropout=dropout, heads=2, flow=flow)

        self.flow = flow
        self.gate_layer = nn.Sequential(
            nn.Linear(2 * feat_size, feat_size, bias=False),
            nn.Sigmoid()
        )
        self.fuse_layer = nn.Sequential(
            nn.Linear(feat_size, feat_size, bias=False),
            nn.Tanh()
        )

    def forward(self, src_nodes, tgt_nodes, edge_index, direction=0):
        """
        nodes: shape [*, node_feat_size]
        edge_index: shape [2, *]
        """
        # when direction = 0
        # source entity => target title
        # query = title
        # key = entity
        x = self.conv1((src_nodes, tgt_nodes), edge_index, direction=direction)
        if direction == 0:
            h = torch.cat([x, tgt_nodes], dim=-1)
        else:
            h = torch.cat([tgt_nodes, x], dim=-1)
        gate = self.gate_layer(h)
        output = gate * self.fuse_layer(x) + (1.0 - gate) * tgt_nodes
        return output
