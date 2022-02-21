# -*- coding: utf-8 -*-
"""Script for building the training examples.

"""
import os
import json
import random
import pickle
import argparse
import multiprocessing as mp
from typing import List, Dict
from numpy.lib.function_base import append

import tqdm
import torch
import pandas as pd
import numpy as np
import networkx as nx

from datasets.vocab import WordVocab
from datasets.graph_datasets import KGData, NewsDataset
from utils.build_util import build_subgraph

random.seed(7)
ROOT_PATH = os.environ["MINDWD"]
PADDING_NEWS = {
    "ents": ['<pad>'],
    "neighbors": [],
    "title": [0, 1]
}


def convert_kgdata(nodes, edges, bi_edges, 
                   hist_mask, pos_mask, neg_masks, 
                   hist_seqs, hist_seq_lens, 
                   pos_seq, pos_seq_len, 
                   neg_seqs, neg_seq_lens,max_hist_len):
    node_features = torch.LongTensor(nodes).unsqueeze(1)
    edge_index = torch.tensor(edges, dtype=torch.short)
    bi_edge_index = torch.tensor(bi_edges, dtype=torch.short)

    ents = node_features
    news = torch.LongTensor(list(range(max_hist_len + 1 + 4))).unsqueeze(1)
    y = torch.LongTensor([0, ])

    data = KGData(ents=ents, news=news, edge_index=edge_index, bi_edge_index=bi_edge_index, y=y)
    data.hist_mask = torch.BoolTensor(hist_mask).unsqueeze(1)
    data.pos_mask = torch.BoolTensor(pos_mask).unsqueeze(1)
    data.neg_masks = torch.BoolTensor(neg_masks).T
    data.hist_seqs = torch.LongTensor(hist_seqs).unsqueeze(0)
    data.hist_seq_lens = torch.CharTensor(hist_seq_lens).unsqueeze(0)
    data.pos_seq = torch.LongTensor([pos_seq, ]).unsqueeze(0)
    data.pos_seq_len = torch.CharTensor([pos_seq_len, ])
    data.neg_seqs = torch.LongTensor(neg_seqs).unsqueeze(0)
    data.neg_seq_lens = torch.CharTensor(neg_seq_lens).unsqueeze(0)

    return data


def build_examples(rank: int,
                   args: argparse.Namespace,
                   df: List[str],
                   graph: nx.DiGraph,
                   ent_vocab: WordVocab,
                   hist_dict: Dict[str, List[str]],
                   news_info: Dict[str, List[str]],
                   output_path: str) -> None:
    """
    Args:
        rank: process id
        args: config
        df: behavior data
        graph: entity graph
        ent_vocab: entity vocab
        hist_dict: uid => newsid list
        news_info: news_id => entity, neighbors, title
        output_path: output path

    Returns:
        None
    """
    random.seed(7)

    def _padding_history(hists: List[str], max_count: int = 10) -> List[str]:
        hists = [x for x in hists if x in news_info]
        hist_len = min(len(hists), max_count)
        if len(hists) < max_count:
            return hists + ["PAD", ] * (max_count - len(hists)), hist_len
        return hists[-max_count:], hist_len

    print("Loading sample data")
    if rank == 0:
        loader = tqdm.tqdm(df, desc="Building")
    else:
        loader = df
    data_list = []
    for row in loader:
        row = json.loads(row)
        uid = row["uid"]

        hist = hist_dict.get(uid, [])
        if len(hist) == 0:
            continue
        sampled_hist, hist_truth_len = _padding_history(hist, args.max_hist_length)
        hist_news_ents = [news_info.get(x, PADDING_NEWS)["ents"] for x in sampled_hist]
        hist_ent_set = set([])
        for news_ents in hist_news_ents:
            hist_ent_set = hist_ent_set.union(set(news_ents))
        hist_neighbors = []
        for x in sampled_hist:
            hist_neighbors += news_info.get(x, PADDING_NEWS)["neighbors"]
        hist_titles = [news_info.get(x, PADDING_NEWS)["title"] for x in sampled_hist]
        hist_news_indices, hist_news_lens = [], []
        for title in hist_titles:
            news_index, length = title
            hist_news_indices.append(news_index)
            hist_news_lens.append(length)
        
        for pair in row["pairs"] :
            news_id = pair[0] # pos news
            if news_id not in news_info:
                continue
            pos_news_ents = [news_info.get(news_id, PADDING_NEWS)["ents"], ]
            pos_ent_set = set(pos_news_ents[0])
            pos_neighbors = news_info.get(news_id, PADDING_NEWS)["neighbors"]
            pos_title = news_info.get(news_id, PADDING_NEWS)["title"]
            pos_news_index, pos_news_len = pos_title

            random_neg_ids = pair[1] # pos news list
            neg_news_ents = [news_info.get(x, PADDING_NEWS)["ents"] for x in random_neg_ids]
            neg_ent_set_list = [set(x) for x in neg_news_ents]
            neg_neighbors = []
            for x in random_neg_ids:
                neg_neighbors += news_info.get(x, PADDING_NEWS)["neighbors"]
            neg_titles = [news_info.get(x, PADDING_NEWS)["title"] for x in random_neg_ids]
            neg_news_indices, neg_news_lens = [], []
            for title in neg_titles:
                news_index, length = title
                neg_news_indices.append(news_index)
                neg_news_lens.append(length)

            all_neighbors = set(hist_neighbors + pos_neighbors + neg_neighbors)
            all_news_ents = hist_news_ents + pos_news_ents + neg_news_ents
            nodes, edge_index, bi_edge_index, hist_mask, pos_mask, neg_mask_list = \
                build_subgraph(graph, ent_vocab, all_news_ents,
                                hist_ent_set, pos_ent_set, neg_ent_set_list, all_neighbors)
            
            data = convert_kgdata(nodes=nodes, edges=edge_index, bi_edges=bi_edge_index, 
                                    hist_mask=hist_mask, pos_mask=pos_mask, neg_masks=neg_mask_list, 
                                    hist_seqs=hist_news_indices, hist_seq_lens=hist_news_lens,
                                    pos_seq=pos_news_index, pos_seq_len=pos_news_len,
                                    neg_seqs=neg_news_indices, neg_seq_lens=neg_news_lens,
                                    max_hist_len=args.max_hist_length)
            data_list.append(data)

    data, slices = NewsDataset.collate(data_list)
    torch.save((data, slices), output_path)


def main(args):
    f_samples = os.path.join(ROOT_PATH, "data", args.fsize, args.fsamples)
    f_hist = os.path.join(ROOT_PATH, "data", args.fsize, args.fhist)
    f_graph = os.path.join(ROOT_PATH, "data", args.fvocab, "graph.bin")
    f_ent_vocab = os.path.join(ROOT_PATH, "data", args.fvocab, "ent_vocab.bin")
    f_news_info = os.path.join(ROOT_PATH, "data", args.fsize, "news_dict-hist50-{}.txt".format(args.fvocab))

    # Load behavior data & split
    # df = pd.read_csv(f_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    df = open(f_samples, "r", encoding="utf-8").readlines()
    subdf_len = len(df) // args.processes
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    # Load user history dict
    hist_dict = dict()
    lines = open(f_hist, "r", encoding="utf8").readlines()
    error_line = 0
    for l in lines:
        row = l.strip().split("\t")
        if len(row) == 1:
            error_line += 1
            continue
        hist_dict[row[0]] = row[1].split(',')
    print("User history error Line: ", error_line)

    # Load news info
    news_info = dict()
    lines = open(f_news_info, "r", encoding="utf8").readlines()
    for line in lines:
        row = line.strip().split("\t")
        news_info[row[0]] = json.loads(row[1])

    # Load two vocabs and graph
    ent_vocab = WordVocab.load_vocab(f_ent_vocab)
    graph = pickle.load(open(f_graph, 'rb'))

    processes = []
    for i in range(args.processes):
        output_path = os.path.join(ROOT_PATH, "data", args.fsize, args.fout, "training_set_dist-{}.pt".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], graph, ent_vocab, hist_dict, news_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--fnews", default="train/news.tsv", type=str,
                        help="Path of the news info file.")
    parser.add_argument("--fsamples", default="train/samples.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="hop1_cocur_bip_hist50", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--fvocab", default="new_vocab_graph", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--fhist", default="train-user_hist_dict.txt", type=str,
                        help="Name of the user history dict.")
    parser.add_argument("--max_hist_length", default=50, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")

    args = parser.parse_args()

    main(args)
