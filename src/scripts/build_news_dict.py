# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/7

Build the map of news id <==> json
Json format：
{
    "ents": ["aaa", "aaa", "aaa"],
    "neighbors": ["bbb", "bbb", "bbb"],
    "title": "cc dd ee ff gg"
}
"""
import os
import json
import random
import pickle
import argparse
from typing import Set, Dict
import numpy as np
import pandas as pd
import networkx as nx
from datasets.vocab import WordVocab
from utils.build_util import word_tokenize

random.seed(7)
ROOT_PATH = os.environ["MINDWD"]


def build_ent_neighbors_dict(all_ents: Set[str], graph: nx.DiGraph, ent_neighbors_dict: Dict[str, Set[str]], max_neighbor_cnt=5) -> Dict[str, Set[str]]:
    """
    For each entity, sampling max_neighbor_cnt neighbors in the entity graph
    Args:
        all_ents: all entities in all news
        graph: the entity graph
        max_neighbor_cnt:

    Returns:
        a dict mapping an ent to a set of its neighbors
    """
    not_found_cnt = 0

    for ent in all_ents:
        if ent in ent_neighbors_dict:
            continue
        if ent not in graph:
            not_found_cnt += 1
            continue
        sorted_neighbors = sorted(dict(graph[ent]).items(), key=lambda item: -1 * item[1]['weight'])
        neighbors = [x[0] for x in sorted_neighbors][:max_neighbor_cnt]
        if len(neighbors) == 0:
            continue
        ent_neighbors_dict[ent] = set(neighbors)

    print("There are {}/{} entities not found in the KG.".format(not_found_cnt, len(all_ents)))

    return ent_neighbors_dict


def build_news_neighbors_dict(news_ents_dict: Dict[str, Set[str]], ent_neighbors_dict: Dict[str, Set[str]], news_neighbors_dict: Dict[str, Set[str]]) \
        -> Dict[str, Set[str]]:
    """
    For each news, sampling self_entity_count * 2 neighbors in the entity graph
    Args:
        news_ents_dict: a dict mapping the news id to self entities
        ent_neighbors_dict: a dict mapping an ent to a set of its neighbors

    Returns:
        a dict mapping an news to a set of its neighbors
    """
    for news_id, ent_set in news_ents_dict.items():
        if news_id in news_neighbors_dict:
            continue
        self_ent_cnt = len(ent_set)
        neighbors = []
        for ent in ent_set:
            neighbors += list(ent_neighbors_dict.get(ent, []))
        neighbors = list(set(neighbors).difference(ent_set))
        neighbors.sort()
        if len(neighbors) <= self_ent_cnt * 2:
            news_neighbors_dict[news_id] = set(neighbors)
            continue
        news_neighbors_dict[news_id] = set(random.sample(neighbors, self_ent_cnt * 2))

    return news_neighbors_dict


def process(news_df, graph, newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict, valid_ent_set=None):
    all_ents = []
    for row in news_df[["newsid", "title_ents", "abs_ents"]].values:
        if row[0] in newsid_ents_dict:
            continue
        ents = [x["WikidataId"] for x in json.loads(row[-2])]
        ents += [x["WikidataId"] for x in json.loads(row[-1])]
        if valid_ent_set is None:
            ents = ents[:15]
        else:
            ents = [x for x in ents if x in valid_ent_set][:15]
        all_ents += ents
        newsid_ents_dict[row[0]] = set(ents)
    all_ents = set(all_ents)

    print("Building entity neighbor dict")
    ent_neighbors_dict = build_ent_neighbors_dict(all_ents, graph, ent_neighbors_dict)
    print("Building news neighbor dict")
    newsid_neighbors_dict = build_news_neighbors_dict(newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict)

    return newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict


def load_real_train_news(f_train_samples, f_train_hist):
    news_list_in_train = []
    lines = open(f_train_samples, "r", encoding="utf-8").readlines()
    for l in lines:
        j = json.loads(l)
        for pair in j["pairs"]:
            news_list_in_train.append(pair[0])
            news_list_in_train += pair[1]
    
    lines = open(f_train_hist, "r").readlines()
    for l in lines:
        news_list_in_train += l.split()[1].split(',')
    news_list_in_train = set(news_list_in_train)
    return news_list_in_train


def main(args):
    f_out = os.path.join(ROOT_PATH, "data", args.fsize, "news_dict-hist50-{}.txt".format(args.fvocab))
    f_title_matrix = os.path.join(ROOT_PATH, "data", args.fsize, "news_title.npy")
    f_graph = os.path.join(ROOT_PATH, "data", args.fvocab, "graph.bin")
    f_train_news = os.path.join(ROOT_PATH, "data", args.fsize, "train/news.tsv")
    f_dev_news = os.path.join(ROOT_PATH, "data", args.fsize, "dev/news.tsv")
    f_test_news = os.path.join(ROOT_PATH, "data", args.fsize, "test/news.tsv")
    f_train_samples = os.path.join(ROOT_PATH, "data", args.fsize, "train/samples.tsv")
    f_train_hist = os.path.join(ROOT_PATH, "data", args.fsize, "train-user_hist_dict.txt")

    print("Loading training news")
    train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                           names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                           quoting=3)
    all_news = train_news.copy(deep=True)
    dev_news = None
    if os.path.exists(f_dev_news):
        print("Loading dev news")
        dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
        all_news = pd.concat([all_news, dev_news], ignore_index=True)
    test_news = None
    if os.path.exists(f_test_news):
        print("Loading testing news")
        test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
                                names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                                quoting=3)
        all_news = pd.concat([all_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates("newsid")
    print("All news: {}".format(len(all_news)))

    # Build news_id => neighbor entity set
    # Load entity vocab and graph
    graph = pickle.load(open(f_graph, 'rb'))

    # Build news_id => title, news_id => cates
    newsid_title_dict = {}
    newsid_cate_dict = {}
    for row in all_news[["newsid", "cate", "subcate", "title"]].values:
        title = " ".join(word_tokenize(row[-1])[:20])
        newsid_title_dict[row[0]] = title
        newsid_cate_dict[row[0]] = [row[1], row[2]]

    news_list_in_train = load_real_train_news(f_train_samples=f_train_samples, f_train_hist=f_train_hist)
    print("News in train: {}/{}".format(len(news_list_in_train), len(train_news)))
    train_news = train_news[train_news.newsid.isin(news_list_in_train)]

    # Build news_id => entity set
    ent_neighbors_dict, newsid_ents_dict, newsid_neighbors_dict = {}, {}, {}
    # Processing training set
    newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict = process(train_news, graph, newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict)

    all_train_neighbors, all_train_ents = [], []
    for x in newsid_neighbors_dict.values():
        all_train_neighbors += list(x)
    for x in newsid_ents_dict.values():
        all_train_ents += list(x)
    
    # all valid ents = train ents + train ent neighbors
    all_ents = set(all_train_ents).union(set(all_train_neighbors))
    print("All valid ent: {}".format(len(all_ents)))
    print("Original Graph: {} nodes".format(len(graph.nodes)))
    graph.remove_nodes_from([n for n in graph if n not in all_ents])
    print("Cleaned Graph: {} nodes".format(len(graph.nodes)))

    # Processing dev and test set
    if dev_news is not None:
        newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict = process(dev_news, graph, newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict, valid_ent_set=all_ents)
    if test_news is not None:
        newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict = process(test_news, graph, newsid_ents_dict, ent_neighbors_dict, newsid_neighbors_dict, valid_ent_set=all_ents)

    # Title matrix: news index => title word indices
    f_word_vocab = os.path.join(ROOT_PATH, "data", args.fvocab, "word_vocab.bin")
    word_vocab = WordVocab.load_vocab(f_word_vocab)
    news2title = np.zeros((len(newsid_title_dict) + 1, args.max_title_len), dtype=int)
    news2nid = {}
    news2nid['PAD'] = 0
    news2title[0], cur_len = word_vocab.to_seq('<pad>', seq_len=args.max_title_len, with_len=True)
    news_index = 1

    with open(f_out, "w", encoding="utf8") as fw:
        default_info = {
            "ents": ['<pad>'],
            "neighbors": [],
            "title": [0, 1]
        }
        fw.write("PAD\t{}\n".format(json.dumps(default_info, ensure_ascii=False)))

        for news_id in newsid_title_dict:
            news2nid[news_id] = news_index
            cur_title = newsid_title_dict.get(news_id, "")
            news2title[news_index], cur_len = word_vocab.to_seq(cur_title, seq_len=args.max_title_len, with_len=True)

            news_info = {
                    "ents": newsid_cate_dict[news_id] + list(newsid_ents_dict.get(news_id, [])),
                    "neighbors": list(newsid_neighbors_dict.get(news_id, [])),
                    "title": [news2nid[news_id], cur_len]
                }
            fw.write("{}\t{}\n".format(news_id, json.dumps(news_info, ensure_ascii=False)))

            news_index += 1
    
    np.save(f_title_matrix, news2title)
    print("title embedding: ", news2title.shape)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--fvocab", default="new_vocab_graph", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--max_title_len", default=20, type=int,
                        help="Max length of the title.")

    args = parser.parse_args()

    main(args)
