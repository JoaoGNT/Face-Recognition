import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import imageProcessor2

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor2.dataMatrix)
print(data)
print(imageProcessor2.labels)

ModelSvm = svm.SVC()

parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [10 ** -4, 1, 10 ** 4], 'degree': [2, 3, 4, 5],
              'gamma': [10 ** -(1 / 2), 1, 10 ** (1 / 2)], 'random_state': [1],
              'class_weight': [{-1: w} for w in [1, 5, 8, 10]]}

clf = GridSearchCV(ModelSvm, parameters, cv=10)
clf.fit(data, imageProcessor2.labels)

dataFrame = pd.DataFrame.from_dict(clf.cv_results_)
dataFrame.to_excel("svmEDistance.xlsx", sheet_name="svm")
print(clf.best_params_)
