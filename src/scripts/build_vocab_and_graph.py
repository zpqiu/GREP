# encoding: utf-8
"""
1. Build entity vocab
2. Build graph
"""

import os
import json
import pickle
import argparse

import pandas as pd
import numpy as np
import networkx as nx

from datasets.vocab import WordVocab
from utils.build_util import word_tokenize

ROOT_PATH = os.environ["MINDWD"]


def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Load Glove embedding
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()
    emb_dict = dict()
    error_line = 0
    embed_size = 0
    for line in lines:
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1:]]
            emb_dict[row[0]] = np.array(embedding)
            if embed_size == 0:
                embed_size = len(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    # embed_size = len(emb_dict.values()[0])
    # build embedding weights for model
    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for i, word in enumerate(vocab.itos):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(embed_size,))
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    np.save(weights_output_file, weights_matrix)


def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])


def build_entire_graph(news_list, vocab):
    edges = list()

    for news in news_list:
        entities = news.split()
        edges += [[entities[0], entities[1], 1], [entities[1], entities[0], 1]]

        for i in range(2, len(entities)):
            for j in range(i + 1, len(entities)):
                if entities[i] == entities[j]:
                    continue
                edges.append([entities[i], entities[j], 1])
                edges.append([entities[j], entities[i], 1])

    edge_df = pd.DataFrame(edges, columns=["from", "to", "weight"])
    edge_weights = edge_df.groupby(["from", "to"]).apply(lambda x: sum(x["weight"]))
    weighted_edges = edge_weights.to_frame().reset_index().values

    dg = nx.DiGraph()
    dg.add_weighted_edges_from(weighted_edges)

    return dg


def main(cfg):
    # Build vocab
    print("Loading news info")
    f_train_news = os.path.join(ROOT_PATH, "data", args.fsize, "train/news.tsv")
    f_dev_news = os.path.join(ROOT_PATH, "data", args.fsize, "dev/news.tsv")
    f_test_news = os.path.join(ROOT_PATH, "data", args.fsize, "test/news.tsv")

    print("Loading training news")
    all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                           names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                           quoting=3)
    if os.path.exists(f_dev_news):
        print("Loading dev news")
        dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
        all_news = pd.concat([all_news, dev_news], ignore_index=True)
    if os.path.exists(f_test_news):
        print("Loading testing news")
        test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
                                names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                                quoting=3)
        all_news = pd.concat([all_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates("newsid")
    print("All news: {}".format(len(all_news)))

    df = all_news
    df = df.fillna(" ")
    df['ents1'] = df['title_ents'].apply(lambda x: parse_ent_list(x))
    df['ents2'] = df['abs_ents'].apply(lambda x: parse_ent_list(x))
    df["ent_list"] = df[["cate", "subcate", "ents1", "ents2"]].apply(lambda x: " ".join(x), axis=1)

    ent_vocab = WordVocab(df.ent_list.values, max_size=cfg.size, min_freq=1, lower=cfg.lower)
    print("ENTITY VOCAB SIZE: {}".format(len(ent_vocab)))
    fpath = os.path.join(ROOT_PATH, cfg.output, "ent_vocab.bin")
    ent_vocab.save_vocab(fpath)

    # Build News Graph
    print("Building Graph")
    graph = build_entire_graph(df.ent_list.values, ent_vocab)
    print("Original Graph built from all news:", len(graph.nodes))

    graph.add_node("<pad>")
    graph_path = os.path.join(ROOT_PATH, cfg.output, "graph.bin")
    pickle.dump(graph, open(graph_path, 'wb'))

    # Building for text
    df['title_token'] = df['title'].apply(lambda x: ' '.join(word_tokenize(x)))
    df['abs_token'] = df['abs'].apply(lambda x: ' '.join(word_tokenize(x)))
    df["text"] = df[["title_token", "abs_token"]].apply(lambda x: " ".join(x), axis=1)

    # Build word vocab
    word_vocab = WordVocab(df.text.values, max_size=cfg.size, min_freq=1, lower=cfg.lower)
    print("TEXT VOCAB SIZE: {}".format(len(word_vocab)))
    f_text_vocab_path = os.path.join(ROOT_PATH, cfg.output, "word_vocab.bin")
    word_vocab.save_vocab(f_text_vocab_path)

    # Build word embeddings
    print("Building word embedding matrix")
    pretrain_path = os.path.join(ROOT_PATH, cfg.pretrain)
    weight_path = os.path.join(ROOT_PATH, cfg.output, "word_embeddings.bin")
    build_word_embeddings(word_vocab, pretrain_path, weight_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--pretrain", default="data/glove.840B.300d.txt", type=str,
                        help="Path of the raw review data file.")

    parser.add_argument("--output", default="data/new_vocab_graph", type=str,
                        help="Path of the training data file.")
    parser.add_argument("--size", default=80000, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--lower", action='store_true')

    args = parser.parse_args()

    main(args)
