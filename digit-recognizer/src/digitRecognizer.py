import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

trainDataset = pd.read_csv('../input/train.csv', sep=',')
testDataset = pd.read_csv('../input/test.csv', sep=',')

x_train, x_val, y_train, y_val = train_test_split(
    trainDataset.iloc[:, 1:], trainDataset.iloc[:, 0], train_size=0.9)

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

y_predict = dtc.predict(x_val)

print(dtc.score(x_val, y_val))

y_predict = dtc.predict(testDataset)

answers = pd.DataFrame({'ImageId': list(range(1, len(y_predict) + 1)),
                        'Label': y_predict})

answers.to_csv('../output/answers.csv', index=False, header=True)
