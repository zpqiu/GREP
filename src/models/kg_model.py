import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.vocab import WordVocab
from modules.title import TitleEncoder
from modules.user import UserEncoder
from modules.attentions import SelfAttend
from modules.gat import GatNet, CrossGatNet
from modules.graph_pooling import MaskGlobalAttention, GateLayer


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # Config
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.ent_embedding_size = cfg.model.ent_embedding_size
        self.conv_hidden_size = cfg.model.conv_hidden_size
        self.neg_count = cfg.model.neg_count
        self.max_news_len = cfg.dataset.max_news_len
        self.max_hist_len = cfg.dataset.max_hist_len

        # Init Layers
        self.ent_vocab = WordVocab.load_vocab(cfg.dataset.ent_vocab)
        self.ent_embedding = nn.Embedding(len(self.ent_vocab), cfg.model.ent_embedding_size)
        self.title_embedding = torch.LongTensor(np.load(cfg.dataset.title_embedding))
        self.dropout = nn.Dropout(cfg.model.dropout)

        # Title view layers
        self.title_encoder = TitleEncoder(cfg)
        self.user_encoder = UserEncoder(cfg)
        self.user_encoder2 = UserEncoder(cfg)
        self.user_encoder3 = UserEncoder(cfg)

        # KG view layers
        self.gnn = GatNet(cfg.model.ent_embedding_size, self.conv_hidden_size)
        self.gnn2 = GatNet(cfg.model.ent_embedding_size, self.conv_hidden_size)

        # Cross layers
        self.cross_gnn = CrossGatNet(cfg.model.ent_embedding_size, self.conv_hidden_size, flow="source_to_target")
        self.cross_gnn2 = CrossGatNet(cfg.model.ent_embedding_size, self.conv_hidden_size, flow="source_to_target")

        # Aggregate & Prediction layers
        self.news_self_attend = SelfAttend(cfg.model.ent_embedding_size)
        self.gate_layer = nn.Linear(cfg.model.ent_embedding_size, 1)
        self.pooling = MaskGlobalAttention(self.gate_layer)

        self.weight_layer = nn.Sequential(
            nn.Linear(cfg.model.ent_embedding_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        # Unpack data instance
        # nodes: [node_num, 1]
        # edge_index: [2, edge_num]
        # y: [batch_size, 1]
        # batch: [node_num, ]
        # hist_mask & pos_mask: [node_num, 1]
        # neg_masks: [node_num, neg_count]
        # hist_seqs: [batch_size, max_hist_len, max_news_len]
        # hist_seq_lens: [batch_size, max_hist_len]
        # pos_seq: [batch_size, max_news_len]
        # pos_seq_len: [batch_size]
        # neg_seqs: [batch_size, neg_count, max_news_lens]
        # neg_seq_lens: [batch_size, neg_count]
        nodes, edge_index, bi_edge_index, y, batch = data.x.long(), data.edge_index.long(), data.bi_edge_index.long(), data.y, data.batch.long()
        hist_mask, pos_mask, neg_masks = data.hist_mask.long(), data.pos_mask.long(), data.neg_masks.long()
        hist_seqs, hist_seq_lens, pos_seq, pos_seq_len, neg_seqs, neg_seq_lens =\
            data.hist_seqs.long(), data.hist_seq_lens.long(), data.pos_seq.long(), data.pos_seq_len.long(), data.neg_seqs.long(), data.neg_seq_lens.long()
        re_bi_edge_index = torch.cat([bi_edge_index[1, :].unsqueeze(0), bi_edge_index[0, :].unsqueeze(0)], dim=0)

        hist_seqs = self.title_embedding[hist_seqs]
        pos_seq = self.title_embedding[pos_seq.squeeze()]
        neg_seqs = self.title_embedding[neg_seqs]
        # title init
        user_title_hidden = self.title_encoder.encode_user(hist_seqs, hist_seq_lens)
        pos_title_hidden = self.title_encoder.encode_news(pos_seq, pos_seq_len)
        neg_title_hiddens = self.title_encoder.encode_news(neg_seqs.view(-1, self.max_news_len),
                                                           neg_seq_lens.view(-1))
        neg_title_hiddens = neg_title_hiddens.view(-1, self.neg_count, self.ent_embedding_size)
        target_title_hiddens = torch.cat([pos_title_hidden.unsqueeze(1), neg_title_hiddens], dim=1)

        # graph init
        node_embeddings = self.ent_embedding(nodes).squeeze(1)
        node_hiddens = node_embeddings

        # title self
        user_title_hidden = self.user_encoder(user_title_hidden)

        # gnn1
        node_hiddens = self.gnn(node_hiddens, edge_index)

        # cross1
        news_hiddens = torch.cat([user_title_hidden, target_title_hiddens], dim=1)
        news_hiddens = news_hiddens.view(-1, self.ent_embedding_size)
        node_cross_hiddens = self.cross_gnn(news_hiddens, node_hiddens, bi_edge_index, direction=1)
        news_cross_hiddens = self.cross_gnn(node_hiddens, news_hiddens, re_bi_edge_index, direction=0)
        news_cross_hiddens = news_cross_hiddens.view(-1, (self.max_hist_len + self.neg_count + 1),
                                                     self.ent_embedding_size)

        # title self2
        user_title_hidden = news_cross_hiddens[:, :self.max_hist_len, :]
        user_title_hidden = self.user_encoder2(user_title_hidden)
        target_title_hiddens = news_cross_hiddens[:, self.max_hist_len:, :]

        # gnn2
        node_hiddens = self.gnn2(node_cross_hiddens, edge_index)

        # cross2
        news_hiddens = torch.cat([user_title_hidden, target_title_hiddens], dim=1)
        news_hiddens = news_hiddens.view(-1, self.ent_embedding_size)
        node_cross_hiddens = self.cross_gnn2(news_hiddens, node_hiddens, bi_edge_index, direction=1)
        news_cross_hiddens = self.cross_gnn2(node_hiddens, news_hiddens, re_bi_edge_index, direction=0)
        news_cross_hiddens = news_cross_hiddens.view(-1, (self.max_hist_len + self.neg_count + 1),
                                                     self.ent_embedding_size)
        node_hiddens = node_cross_hiddens

        # title self3
        user_title_hidden = news_cross_hiddens[:, :self.max_hist_len, :]
        user_title_hidden = self.user_encoder3(user_title_hidden)
        target_title_hiddens = news_cross_hiddens[:, self.max_hist_len:, :]

        # title attend
        user_title_hidden = self.news_self_attend(user_title_hidden)
        user_final_title = user_title_hidden.repeat(1, self.neg_count + 1).view(-1, self.ent_embedding_size)
        target_final_titles = target_title_hiddens.reshape(-1, self.ent_embedding_size)

        all_masks = torch.cat([hist_mask, pos_mask, neg_masks], dim=1)
        # [batch_size, neg_count+2, ent_embedding_size]
        pooled_hiddens = self.pooling(x=node_hiddens, batch=batch, mask=all_masks)

        user_final_graph = pooled_hiddens[:, 0, :].repeat(1, self.neg_count + 1).view(-1, self.ent_embedding_size)
        target_final_graphs = pooled_hiddens[:, 1:, :].reshape(-1, self.ent_embedding_size)

        # PREDICTION
        user_weight = self.weight_layer(torch.cat([user_final_graph, user_final_title], dim=-1))
        user_hiddens = user_weight * user_final_graph + (1.0-user_weight) * user_final_title
        target_weight = self.weight_layer(torch.cat([target_final_graphs, target_final_titles], dim=-1))
        target_hiddens = target_weight * target_final_graphs + (1.0-target_weight) * target_final_titles

        logits = torch.sum(user_hiddens * target_hiddens, dim=-1)
        # logits = self.prediction_layer(torch.cat([user_hiddens, target_hiddens], dim=-1))
        logits = logits.view(-1, self.neg_count + 1)

        return logits

    def training_step(self, data):
        # REQUIRED
        logits = self.forward(data)

        target = data.y
        loss = F.cross_entropy(logits, target)

        return loss

    def predict(self, data):
        # Unpack data instance
        # nodes: [node_num, 1]
        # edge_index: [2, edge_num]
        # batch: [node_num, ]
        # hist_mask: [node_num, 1]
        # target_mask: [node_num, 1]
        # hist_seqs: [batch_size, max_hist_len, max_news_len]
        # hist_seq_lens: [batch_size, max_hist_len]
        # target_seq: [batch_size, max_news_len]
        # target_seq_len: [batch_size]
        nodes, edge_index, bi_edge_index, batch = data.x.long(), data.edge_index.long(), data.bi_edge_index.long(), data.batch.long()
        hist_mask, target_mask = data.hist_mask.long(), data.target_mask.long()
        hist_seqs, hist_seq_lens, target_seq, target_seq_len = \
            data.hist_seqs.long(), data.hist_seq_lens.long(), data.target_seq.long(), data.target_seq_len.long()
        re_bi_edge_index = torch.cat([bi_edge_index[1, :].unsqueeze(0), bi_edge_index[0, :].unsqueeze(0)], dim=0)

        hist_seqs = self.title_embedding[hist_seqs]
        target_seq = self.title_embedding[target_seq.squeeze()]
        # title init
        if target_seq.dim() < 2:
            target_seq = target_seq.unsqueeze(0)
        if hist_seqs.dim() < 3:
            hist_seqs = hist_seqs.unsqueeze(0)
        user_title_hidden = self.title_encoder.encode_user(hist_seqs, hist_seq_lens)
        target_title_hiddens = self.title_encoder.encode_news(target_seq, target_seq_len)

        # graph init
        node_embeddings = self.ent_embedding(nodes).squeeze(1)
        node_hiddens = node_embeddings

        # title self1
        user_title_hidden = self.user_encoder(user_title_hidden)

        # gnn1
        node_hiddens = self.gnn(node_hiddens, edge_index)

        # cross1
        news_hiddens = torch.cat([user_title_hidden, target_title_hiddens.unsqueeze(1)], dim=1)
        news_hiddens = news_hiddens.view(-1, self.ent_embedding_size)
        node_cross_hiddens = self.cross_gnn(news_hiddens, node_hiddens, bi_edge_index, direction=1)
        news_cross_hiddens = self.cross_gnn(node_hiddens, news_hiddens, re_bi_edge_index, direction=0)
        news_cross_hiddens = news_cross_hiddens.view(-1, (self.max_hist_len + 1),
                                                     self.ent_embedding_size)

        # title self2
        user_title_hidden = news_cross_hiddens[:, :self.max_hist_len, :]
        user_title_hidden = self.user_encoder2(user_title_hidden)
        target_title_hiddens = news_cross_hiddens[:, self.max_hist_len:, :]

        # gnn2
        node_hiddens = self.gnn2(node_cross_hiddens, edge_index)

        # cross2
        news_hiddens = torch.cat([user_title_hidden, target_title_hiddens], dim=1)
        news_hiddens = news_hiddens.view(-1, self.ent_embedding_size)
        node_cross_hiddens = self.cross_gnn2(news_hiddens, node_hiddens, bi_edge_index, direction=1)
        news_cross_hiddens = self.cross_gnn2(node_hiddens, news_hiddens, re_bi_edge_index, direction=0)
        news_cross_hiddens = news_cross_hiddens.view(-1, (self.max_hist_len + 1),
                                                     self.ent_embedding_size)
        node_hiddens = node_cross_hiddens

        # title self3
        user_title_hidden = news_cross_hiddens[:, :self.max_hist_len, :]
        user_title_hidden = self.user_encoder3(user_title_hidden)
        target_title_hiddens = news_cross_hiddens[:, self.max_hist_len:, :]

        user_title_hidden = self.news_self_attend(user_title_hidden)
        target_title_hiddens = target_title_hiddens.reshape(-1, self.ent_embedding_size)

        all_masks = torch.cat([hist_mask, target_mask], dim=1)
        # [batch_size, 2, ent_embedding_size]
        pooled_hiddens = self.pooling(x=node_hiddens, batch=batch, mask=all_masks)

        user_graph_hidden = pooled_hiddens[:, 0, :]
        target_graph_hiddens = pooled_hiddens[:, 1, :]

        # PREDICTION
        user_weight = self.weight_layer(torch.cat([user_graph_hidden, user_title_hidden], dim=-1))
        user_hiddens = user_weight * user_graph_hidden + (1.0-user_weight) * user_title_hidden
        target_weight = self.weight_layer(torch.cat([target_graph_hiddens, target_title_hiddens], dim=-1))
        target_hiddens = target_weight * target_graph_hiddens + (1.0-target_weight) * target_title_hiddens

        logits = torch.sum(user_hiddens * target_hiddens, dim=-1)
        # logits = self.prediction_layer(torch.cat([user_hiddens, target_hiddens], dim=-1)).squeeze(1)
        # print(logits)

        return logits

    def validation_step(self, data):
        # OPTIONAL
        preds = self.predict(data)

        return preds


class FusionLayer(nn.Module):
    def __init__(self, input_size):
        super(FusionLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer3(self.layer2(self.layer1(input)))
