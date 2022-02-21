# encoding: utf-8
"""
Split the big txt file to multiple parts
Date: 15 Mar, 2019
"""
import os
import random
import argparse

random.seed(7)
ROOT_PATH = os.environ["MINDWD"]

def build(args):
    print("Loading...")
    f_samples = os.path.join(ROOT_PATH, "data/L/test/behaviors.tsv")
    lines = open(f_samples, "r", encoding="utf8").readlines()

    print("There are {0} lines.".format(len(lines)))
    sub_file_length = len(lines) // args.num

    for i in range(args.num):
        output_file_path = os.path.join(ROOT_PATH, "data/L/test/behaviors.p{}.tsv".format(i))
        st = i * sub_file_length
        if i == args.num-1:
            ed = len(lines)
        else:
            ed = st + sub_file_length
        print("Creating sub-file {0} ...".format(i))
        with open(output_file_path, "w", encoding="utf8") as fw:
            for j in range(st, ed):
                fw.write(lines[j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num", type=int, default=7,
                        help="the number of slice. Default 7.")
    args = parser.parse_args()

    build(args)
