# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/10

Some attention layers
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 200),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(200, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output


def create_mask_from_lengths_for_seqs(
        seq_lens: torch.Tensor, max_len: int
) -> torch.Tensor:
    """
    :param seq_lens: shape [batch_size, ]
    :param max_len: int
    :return: shape [batch_size, seq_length]
    """
    segs = torch.arange(max_len, device=seq_lens.device, dtype=seq_lens.dtype).expand(
        len(seq_lens), max_len
    ) < seq_lens.unsqueeze(1)

    return segs.long()
