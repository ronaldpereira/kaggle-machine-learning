#!/home/ronald/dev/kaggle-machine-learning/virtualenv/bin/python3

import pandas as pd


class titanicDataset:
    def __init__(self):
        self.train = pd.read_csv('../input/train.csv', sep=',')
        self.test = pd.read_csv('../input/test.csv', sep=',')
