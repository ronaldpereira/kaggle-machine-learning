#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
import csv

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


def mountTitanicSet(setArray, type = 'train'):
    pclassArray = setArray.loc[:, 'Pclass']

    sexArray = list(map(lambda sex: 0 if str.lower(sex) ==
                        'male' else 1, setArray.loc[:, 'Sex']))

    ageArray = list(map(lambda age: age/100 if not np.isnan(age)
                        else np.nanmean(setArray.loc[:, 'Age'] / 100), setArray.loc[:, 'Age']))

    sibSpParchArray = setArray.loc[:, 'SibSp'] + setArray.loc[:, 'Parch']

    fareArray = list(map(lambda fare: fare if str.lower(str(fare)) != 'nan' else 0, setArray.loc[:, 'Fare']))

    hasCabinArray = list(map(lambda cabin: 1 if str.lower(
        str(cabin)) != 'nan' else 0, setArray.loc[:, 'Cabin']))

    portsArray = ['nan', 'c', 'q', 's']

    embarkGroupArray = list(map(lambda port: portsArray.index(
        str.lower(str(port))), setArray.loc[:, 'Embarked']))

    if type == 'train':
        survivedArray = setArray.loc[:, 'Survived']

        trainingMatrix = np.array([pclassArray, sexArray, ageArray,
                                   sibSpParchArray, fareArray, hasCabinArray, embarkGroupArray])
        
        return np.transpose(trainingMatrix), survivedArray

    elif type == 'test':
        trainingMatrix = np.array([pclassArray, sexArray, ageArray,
                                   sibSpParchArray, fareArray, hasCabinArray, embarkGroupArray])

        return np.transpose(trainingMatrix)

np.set_printoptions(threshold=np.inf)
trainSet = pd.read_csv('../input/train.csv', sep=',')
testSet = pd.read_csv('../input/test.csv', sep=',')

x_train, y_train = mountTitanicSet(trainSet, 'train')

x_test = mountTitanicSet(testSet, 'test')

# x_train, x_test, y_train, y_test = train_test_split(x, y.astype('int'), train_size=0.8)

logReg = LogisticRegression()

logReg.fit(x_train, y_train)

y_predict = logReg.predict(x_test)

# print(f1_score(y_test, y_predict))

csvWriter = csv.writer(open('../output/answers.csv', 'w', newline=''), delimiter=',')

csvWriter.writerow(['PassengerId', 'Survived'])

passengerID = 892
for pred in y_predict:
    csvWriter.writerow([passengerID, pred])
    passengerID += 1
