# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/2

Some help functions for building dataset
"""
import os
import re
import json
from typing import List, Dict, Set

import pandas as pd
import networkx as nx

from datasets.vocab import WordVocab

PADDING_NEWS = "<pad>"
ROOT_PATH = os.environ["MINDWD"]


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def build_subgraph(graph: nx.DiGraph,
                   vocab: WordVocab,
                   news_ents: List[List[str]],
                   hist_ents: Set[str],
                   pos_ents: Set[str],
                   neg_ent_list: List[Set[str]] = None,
                   neighbor_ents: Set[str] = None):
    """
    Args:
        graph: entity graph
        vocab: entity vocab
        news_ents: entities in each news
        hist_ents: entities in user history
        pos_ents: entities in positive news
        neg_ent_list: entities in multiple negative news
        neighbor_ents: neighbor entities of all news

    Returns:
        nodes: List[int], all node ids of these entities
        edge_index: List[List[int]], sparse adj matrix
        bi_edge_index: List[List[int]], news <=> ent bipartite graph edges
        hist_mask: List[int], the list of labels which mask the nodes not in user history
        pos_mask: List[int], the list of labels which mask the nodes not in positive news
        neg_mask_list: List[List[int]], the list of labels which mask the nodes not in negative news
    """
    all_ents = set([])
    all_ents = all_ents.union(hist_ents)
    all_ents = all_ents.union(pos_ents)
    if neg_ent_list is not None:
        for neg_ents in neg_ent_list:
            all_ents = all_ents.union(neg_ents)
    if neighbor_ents is not None:
        all_ents = all_ents.union(neighbor_ents)

    all_ents = list(all_ents)
    sub_graph = graph.subgraph(all_ents)

    impression_ent_id_map = {ent: idx for idx, ent in enumerate(all_ents)}
    sub_graph = nx.relabel_nodes(sub_graph, impression_ent_id_map)

    nodes = vocab.to_seq(all_ents)

    edges = list(sub_graph.edges)
    source_nodes, target_nodes = [], []
    for edge in edges:
        source_nodes.append(edge[0])
        target_nodes.append(edge[1])
    edge_index = [source_nodes, target_nodes]

    bi_src_nodes, bi_tgt_nodes = [], []
    for news_index, ents in enumerate(news_ents):
        for ent in ents[2:]:
            bi_src_nodes.append(news_index)
            bi_tgt_nodes.append(impression_ent_id_map[ent])
    bi_edge_index = [bi_src_nodes, bi_tgt_nodes]

    hist_mask = [1 if x in hist_ents else 0 for x in all_ents]
    pos_mask = [1 if x in pos_ents else 0 for x in all_ents]
    neg_mask_list = []
    if neg_ent_list is not None:
        for neg_ents in neg_ent_list:
            neg_mask_list.append([1 if x in neg_ents else 0 for x in all_ents])

    return nodes, edge_index, bi_edge_index, hist_mask, pos_mask, neg_mask_list
