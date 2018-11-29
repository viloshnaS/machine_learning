import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import csv

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


forest_X = data

forest_X_train, forest_X_test, forest_y_train, forest_y_test= train_test_split(data,targets,test_size=0.25,random_state=0)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(forest_X_train, forest_y_train)

# Make predictions using the testing set
forest_y_pred = regr.predict(forest_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(forest_y_test, forest_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(forest_y_test, forest_y_pred))
print (regr.score(forest_X_test, forest_y_test))
print (regr.score(forest_X_train, forest_y_train))

