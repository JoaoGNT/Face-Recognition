import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import imageProcessor

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(imageProcessor.dataMatrix)

ModelAnn = MLPClassifier()
parameters = {'solver': ('lbfgs', 'adam'), 'alpha': [1e-5, 1, 1e5],
              'hidden_layer_sizes': [(25, 50, 10), (50, 50, 10), (100, 100, 10)]}
ann = GridSearchCV(ModelAnn, parameters, cv=10)
ann.fit(data, imageProcessor.labels)

dataFrame = pd.DataFrame.from_dict(ann.cv_results_)
dataFrame.to_excel("resultsAnn.xlsx", sheet_name="ann")

'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,100,10), random_state=1)
scores = cross_val_score(clf, data , imageProcessor.imageProcessor.labels, cv=9)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)
print(clf.get_params(MLPClassifier).keys())
'''
