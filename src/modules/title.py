# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.vocab import WordVocab
from utils.model_util import build_embedding_layer
from modules.attentions import SelfAttend


class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        self.max_news_len = cfg.dataset.max_news_len
        self.max_hist_len = cfg.dataset.max_hist_len
        self.batch_size = cfg.training.batch_size
        self.hidden_size = cfg.model.hidden_size

        self.vocab = WordVocab.load_vocab(cfg.dataset.word_vocab)
        self.word_embedding = build_embedding_layer(
            pretrained_embedding_path=cfg.dataset.get("word_embedding", ""),
            vocab=self.vocab,
            embedding_dim=cfg.model.word_embedding_size,
        )
        # NRMS hidden size is 150
        self.mh_self_attn = nn.MultiheadAttention(
            self.hidden_size, num_heads=cfg.model.transformer.heads_num
        )
        self.word_self_attend = SelfAttend(self.hidden_size)

        self.user_mh_self_attn = nn.MultiheadAttention(
            self.hidden_size, num_heads=cfg.model.transformer.heads_num
        )
        self.news_self_attend = SelfAttend(self.hidden_size)

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.word_layer_norm = nn.LayerNorm(self.hidden_size)
        self.user_layer_norm = nn.LayerNorm(self.hidden_size)

    def _extract_hidden_rep(self, seqs, seq_lens):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)
        # output = self.word_proj(output)

        return self.word_layer_norm(output + X)

    def encode_news(
            self,
            seqs,
            seq_lens,
    ):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs, seq_lens)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens)

        return self_attend

    def encode_user(self, seqs, seq_lens):
        """

        Args:
            seqs: [*, max_hist_len, max_news_len]
            seq_lens: [*, max_hist_len]

        Returns:
            [*, max_hist_len, hidden_size]
        """
        hiddens = self.encode_news(seqs.view(-1, self.max_news_len), seq_lens.view(-1))

        # [*, max_hist_len, hidden_size]
        hiddens = hiddens.view(-1, self.max_hist_len, self.hidden_size)

        return hiddens
