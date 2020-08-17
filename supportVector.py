import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import imageProcessor

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor.dataMatrix)
ModelSvm = svm.SVC()
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [10 ** -6, 10, 10 ** 6], 'degree': [3, 5],
              'gamma': [10 ** -3, 10, 10 ** 3]}
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
