import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree
import csv
import graphviz
import pydot
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# Create a random dataset

data = []
targets = []
headers = []
months = ["jan", "feb", "mar", "apr","may","jun","jul","aug","sep","oct","nov","dec"]
days = ["mon","tue","wed","thu","fri","sat","sun"]
with open("forestfires.csv") as csvfile:
    forestreader = csv.reader(csvfile, delimiter=',')
    headers = forestreader.next()
    for row in forestreader:
        if len(row) == 13:
            for i in range(2):
                row[i] = int(row[i])
            for i in range(2,4):
                for j in range(len(months)):
                    if row[i] == months[j]:
                        row[i] = int(j)
                    else:
                        for j in range(len(days)):
                            if row[i] == days[j]:
                                row[i] = int(j)
            for i in range(4, 13):
                row[i] = float(row[i])

            targets.append(float(row[12]))
            del row[-1]
            data.append(row)

data = np.array(data)
n_samples = len(data)
n_features = len(data[0])
print('Number of samples:', n_samples)
print('Number of features:', n_features)

del headers[-1]



X = data
y = targets
X_train, X_test, Y_train, Y_test= train_test_split(data,targets,test_size=0.25,random_state=0)

clf_regressor = DecisionTreeRegressor(max_features=3)
clf_regressor .fit(X_train,Y_train)



dot_data = tree.export_graphviz(clf_regressor, out_file="forest.dot",
                         feature_names=headers,
                         filled=True, rounded=True,
                         special_characters=False)
(graph,) = pydot.graph_from_dot_file('forest.dot')
graph.write_png('forest.png')

print(clf_regressor.score(X_train,Y_train))
print(clf_regressor.score(X_test,Y_test))


