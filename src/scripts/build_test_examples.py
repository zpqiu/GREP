# -*- coding: utf-8 -*-
"""Script for building the validation examples.

"""
import os
import json
import random
import pickle
import argparse
import multiprocessing as mp
from typing import List, Dict

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
                   hist_mask, target_mask,
                   hist_seqs, hist_seq_lens, 
                   target_seq, target_seq_len, 
                   label, imp_id, max_hist_len):
    node_features = torch.LongTensor(nodes).unsqueeze(1)
    edge_index = torch.tensor(edges, dtype=torch.short)
    bi_edge_index = torch.tensor(bi_edges, dtype=torch.short)

    ents = node_features
    news = torch.LongTensor(list(range(max_hist_len + 1))).unsqueeze(1)
    y = torch.LongTensor([int(label), ])

    data = KGData(ents=ents, news=news, edge_index=edge_index, bi_edge_index=bi_edge_index, y=y)
    data.hist_mask = torch.BoolTensor(hist_mask).unsqueeze(1)
    data.target_mask = torch.BoolTensor(target_mask).unsqueeze(1)
    data.hist_seqs = torch.LongTensor(hist_seqs).unsqueeze(0)
    data.hist_seq_lens = torch.CharTensor(hist_seq_lens).unsqueeze(0)
    data.target_seq = torch.LongTensor([target_seq, ]).unsqueeze(0)
    data.target_seq_len = torch.CharTensor([target_seq_len, ])
    data.imp_id = torch.LongTensor([imp_id, ])

    return data


def build_examples(rank: int, 
                   args: argparse.Namespace,
                   df: pd.DataFrame,
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
        word_vocab: word vocab
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
    df = df.fillna('')
    
    with open(output_path, "w", encoding="utf-8") as fw:
        if rank == 0:
            loader = tqdm.tqdm(df[["uid", "hist", "imp", "id"]].values, desc="Building")
        else:
            loader = df[["uid", "hist", "imp", "id"]].values
        data_list = []

        for row in loader:
            uid = row[0]

            hist = hist_dict.get(uid, [])
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

            samples = row[2].strip().split()
            for sample in samples:
                news_id = sample

                pos_news_ents = [news_info.get(news_id, PADDING_NEWS)["ents"], ]
                pos_ent_set = set(pos_news_ents[0])
                pos_neighbors = news_info.get(news_id, PADDING_NEWS)["neighbors"]
                pos_title = news_info.get(news_id, PADDING_NEWS)["title"]
                pos_news_index, pos_news_len = pos_title

                all_neighbors = set(hist_neighbors + pos_neighbors)
                all_news_ents = hist_news_ents + pos_news_ents
                nodes, edge_index, bi_edge_index, hist_mask, pos_mask, _ = \
                    build_subgraph(graph, ent_vocab, all_news_ents,
                                   hist_ent_set, pos_ent_set, neighbor_ents=all_neighbors)
                data = convert_kgdata(nodes=nodes, edges=edge_index, bi_edges=bi_edge_index, 
                                      hist_mask=hist_mask, target_mask=pos_mask,
                                      hist_seqs=hist_news_indices, hist_seq_lens=hist_news_lens,
                                      target_seq=pos_news_index, target_seq_len=pos_news_len,
                                      label=0, imp_id=row[-1], max_hist_len=args.max_hist_length)
                data_list.append(data)
        data, slices = NewsDataset.collate(data_list)
        torch.save((data, slices), output_path)


def main(args):
    f_samples = os.path.join(ROOT_PATH, "data", args.fsize, "test/behaviors.{}.tsv".format(args.fsplit))
    f_hist = os.path.join(ROOT_PATH, "data", args.fsize, args.fhist)
    f_graph = os.path.join(ROOT_PATH, "data", args.fvocab, "graph.bin")
    f_ent_vocab = os.path.join(ROOT_PATH, "data", args.fvocab, "ent_vocab.bin")
    f_news_info = os.path.join(ROOT_PATH, "data", args.fsize, "news_dict-hist50-{}.txt".format(args.fvocab))

    # Load behavior data & split
    df = pd.read_csv(f_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
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
        output_path = os.path.join(ROOT_PATH, "data", args.fsize, args.fout, "{}/test_set_dist-{}.pt".format(args.fsplit, i))
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
    parser.add_argument("--fnews", default="test/news.tsv", type=str,
                        help="Path of the news info file.")
    parser.add_argument("--fsplit", default="all", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="hop1_cocur_bip_hist50", type=str,
                        help="Path of the output file.")
    parser.add_argument("--fvocab", default="new_vocab_graph", type=str,
                        help="Path of the entity dir.")
    parser.add_argument("--fhist", default="test-user_hist_dict.txt", type=str,
                        help="Name of the user history dict.")
    parser.add_argument("--max_hist_length", default=50, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--max_title_len", default=20, type=int,
                        help="Max length of the title.")
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")

    args = parser.parse_args()

    main(args)
