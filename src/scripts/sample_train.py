# -*- coding: utf-8 -*-
"""
Build training samples, each sample is a 2-tuple
uid, [(pos_news_id, some neg_news_id), ...]
"""
import os
import json
import random
import argparse

import tqdm
import pandas as pd

random.seed(7)
ROOT_PATH = os.environ["MINDWD"]


def main(args):
    f_behaviors = os.path.join(ROOT_PATH, "data", args.fsize, "train/behaviors.tsv")
    f_out = os.path.join(ROOT_PATH, "data", args.fsize, "train/samples.tsv")

    df = pd.read_csv(f_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    with open(f_out, "w", encoding="utf-8") as fw:
        for row in tqdm.tqdm(df[["uid", "imp"]].values, desc="Building"):
            uid = row[0]
            samples = row[1].strip().split()
            pos_news_ids, neg_news_ids = list(), list()
            for sample in samples:
                news_id, label = sample.split("-")[:2]
                if label == "1":
                    pos_news_ids.append(news_id)
                else:
                    neg_news_ids.append(news_id)

            if len(neg_news_ids) < args.neg_count:
                neg_news_ids += ["PAD", ] * (args.neg_count - len(neg_news_ids))

            train_pairs = []
            for pos_news_id in pos_news_ids:
                random_neg_ids = random.sample(neg_news_ids, args.neg_count)
                train_pairs.append((pos_news_id, random_neg_ids))
            
            j = {
                "uid": uid,
                "pairs": train_pairs
            }
            fw.write(json.dumps(j, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="S", type=str,
                        help="Corpus size")
    parser.add_argument("--neg_count", default=4, type=int,
                        help="Max neg samples according to one pos sample.")

    args = parser.parse_args()

    main(args)
