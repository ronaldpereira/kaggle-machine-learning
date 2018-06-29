#!/home/ronald/dev/kaggle-machine-learning/virtualenv/bin/python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from subprocess import check_output
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.head())
