import pandas as pd
import numpy as np
import scipy.stats as ss

lines = open('all_pred.result', 'r').readlines()

group_preds = {}

for l in lines:
    row = l.strip().split('\t')
    if row[0] not in group_preds:
        group_preds[row[0]] = []
    group_preds[row[0]].append(float(row[-1]))

with open('prediction.txt', 'w') as fw:
    for k in group_preds:
        rank = ss.rankdata(-np.array(group_preds[k])).astype(int).tolist()
        rank_str = '[' + ','.join(list(map(str, rank))) + ']'
        fw.write("{} {}\n".format(k, rank_str))

df = pd.read_csv('prediction.txt', sep=' ', names=['impid','rk'])
df = df.sort_values(by='impid', ascending=True)
df.to_csv('prediction.txt', sep=' ', index=0, index_label=0, header=0)
