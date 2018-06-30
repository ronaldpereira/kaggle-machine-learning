#!/home/ronald/dev/kaggle-machine-learning/virtualenv/bin/python3

import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)

import titanicDataset

titanic_dataset = titanicDataset.titanicDataset()

print(titanic_dataset.train.head(3))
print(titanic_dataset.test.head(3))
