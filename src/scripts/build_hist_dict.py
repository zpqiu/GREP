# -*- encoding:utf-8 -*-
"""
Date: create at 2020/10/2

Build the map of uin <==> click history
"""
import os
import argparse
from typing import List, Dict

import pandas as pd

ROOT_PATH = os.environ["MINDWD"]


def sample_user_history(hists: List[str], max_count: int = 10) -> List[str]:
    """
    Args:
        hists: the complete user history news list
        max_count: the maximum history length accepted by the model

    Returns:
        the sampled user history news list
    """
    return hists[-max_count:]


def build_hist_dict(args: argparse.Namespace,
                    df: pd.DataFrame,
                    user_hist_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Args:
        args: config
        df: user behavior data
        user_hist_dict: existed dict which map uid to history news list

    Returns:
        a new dict
    """
    for uid, hist in df[["uid", "hist"]].values:
        if uid in user_hist_dict:
            continue

        hist = str(hist).strip().split()
        if len(hist) == 0:
            continue

        sampled_hist = sample_user_history(hist, args.max_hist_length)
        user_hist_dict[uid] = sampled_hist
    return user_hist_dict


def main(args):
    behavior_path = os.path.join(ROOT_PATH, "data", args.fsize, args.fmark, "behaviors.tsv")
    out_path = os.path.join(ROOT_PATH, "data", args.fsize, args.fmark + "-" + args.fname)

    user_hist_dict = {}
    print("Building from {}".format(behavior_path))
    train_df = pd.read_csv(behavior_path, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    train_df = train_df[train_df["hist"].isna() == False]
    user_hist_dict = build_hist_dict(args, train_df, user_hist_dict)

    print("User Count: {}".format(len(user_hist_dict)))
    with open(out_path, "w", encoding="utf8") as fw:
        for uid, hist in user_hist_dict.items():
            fw.write("{}\t{}\n".format(uid, ','.join(hist)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--fmark", default="train", type=str,
                        help="train or dev or test")
    parser.add_argument("--fname", default="user_hist_dict.txt", type=str,
                        help="Output file name.")
    parser.add_argument("--max_hist_length", default=50, type=int,
                        help="Max length of the click history of the user.")

    args = parser.parse_args()

    main(args)
