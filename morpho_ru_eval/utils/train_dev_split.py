__author__ = 'Oleg'

import argparse
import numpy as np
from morpho_ru_eval.utils.datasets import read_data


def write_data(file,lst):
    f = open (file,"w",encoding='utf-8')
    for sent in lst:
        f.writelines(sent)
        f.write('\n')
    f.close()


parser = argparse.ArgumentParser(description='split a file into dev/train, proportion 30/70')
parser.add_argument('src', type=str,help='path to source')
parser.add_argument('--train', default="train_set.txt", type=str,help='path to train set')
parser.add_argument('--dev', default="dev_set.txt", type=str,help='path to dev set')
args = parser.parse_args()


sent = read_data(args.src)
sent = np.array(sent)
np.random.shuffle(sent)
idx = round(len(sent)*0.3)

train = sent[idx:]
dev = sent[:idx]

write_data(args.train,train)
write_data(args.dev,dev)