from __future__ import division
import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import logging,sys
from sklearn import datasets
from sklearn import tree
import graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Load the wine dataset
wine = datasets.load_wine()

feature = 7


X=wine.data
y=wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=y)

print(wine.feature_names)
clf = tree.DecisionTreeClassifier()
clf .fit(X_train,y_train)
dot_data = tree.export_graphviz(clf, out_file="wine",
                         feature_names=wine.feature_names,
                         class_names=wine.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
(graph,) = pydot.graph_from_dot_file('wine.dot')
graph.write_png('wine.png')
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
predicted_clf = clf.predict(X_test)

idx = 0
true = 0
false = 0
for i in X_test:

    if predicted_clf[idx]==y_test[idx]:
        true +=1
    else:
        false +=1
    idx +=1

accuracy =  (true/(true+false))*100
print("Positive Class: "+str(true))
print("Negative Class: "+str(false))
print("Accuracy: "+str(accuracy))



