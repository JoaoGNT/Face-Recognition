import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import GridSearchCV

import imageProcessor2

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor2.dataMatrix)
print(data)
print(imageProcessor2.labels)
ModelTree = tree.DecisionTreeClassifier()
parameters = {'criterion': ('gini', 'entropy'), 'class_weight': [{-1: w} for w in [1, 5, 8, 10]]}

clf = GridSearchCV(ModelTree, parameters, cv=10)
clf.fit(data, imageProcessor2.labels)

dataFrame = pd.DataFrame.from_dict(clf.cv_results_)
dataFrame.to_excel("dtreeEDistance.xlsx", sheet_name="dtree")
print(clf.best_params_)
