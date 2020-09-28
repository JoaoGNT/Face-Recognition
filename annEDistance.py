import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import imageProcessor2

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor2.dataMatrix)
print(data)
print(imageProcessor2.labels)
ModelAnn = MLPClassifier()
parameters = {'solver': ['adam'], 'alpha': [1e-5, 1e-3, 1e-1],
              'hidden_layer_sizes': [(25, 50, 10), (50, 50, 10), (100, 100, 10), (100, 50, 10), (50, 10)],
              'random_state': [1], 'activation': ['identity', 'relu', 'logistic']}
clf = GridSearchCV(ModelAnn, parameters, cv=10)
clf.fit(data, imageProcessor2.labels)

dataFrame = pd.DataFrame.from_dict(clf.cv_results_)
dataFrame.to_excel("annEDistance.xlsx", sheet_name="ann")
print(clf.best_params_)
