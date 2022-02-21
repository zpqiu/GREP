# -*- encoding:utf-8 -*-
"""
Date: create at 2020/12/9


"""
import torch.nn as nn


class UserEncoder(nn.Module):
    def __init__(self, cfg):
        super(UserEncoder, self).__init__()
        self.cfg = cfg
        self.max_hist_len = cfg.dataset.max_hist_len
        self.hidden_size = cfg.model.hidden_size

        self.user_mh_self_attn = nn.MultiheadAttention(
            self.hidden_size, num_heads=cfg.model.transformer.heads_num
        )

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.user_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, hiddens):
        """

        Args:
            hiddens: [*, max_hist_len, hidden_size]

        Returns:
            [*, max_hist_len, hidden_size]
        """
        # [*, max_hist_len, hidden_size]
        hiddens = hiddens.view(-1, self.max_hist_len, self.hidden_size)
        hiddens = hiddens.permute(1, 0, 2)

        user_hiddens, _ = self.user_mh_self_attn(hiddens, hiddens, hiddens)
        # [*, hidden_size]
        user_hiddens = user_hiddens.permute(1, 0, 2)
        hiddens = hiddens.permute(1, 0, 2)

        return self.user_layer_norm(user_hiddens + hiddens)
