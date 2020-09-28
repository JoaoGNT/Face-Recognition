import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import imageProcessor

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor.dataMatrix)
print(data)
print(imageProcessor.labels)

ModelSvm = svm.SVC()
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [10 ** -6, 1, 10 ** 6], 'degree': [2, 3, 4, 5],
              'gamma': [10 ** -3, 10, 10 ** 3], 'random_state': [1]}
clf = GridSearchCV(ModelSvm, parameters, cv=10)
clf.fit(data, imageProcessor.labels)

dataFrame = pd.DataFrame.from_dict(clf.cv_results_)
dataFrame.to_excel("resultsSvm.xlsx", sheet_name="svm")
print(clf.best_params_)

"""
evaluationMetrics = cross_validate(clf, data, imageProcessor.labels, cv=9)
predictions = cross_val_predict(clf,data,imageProcessor.labels,cv=9)
print(sorted(evaluationMetrics.keys()))
"""
