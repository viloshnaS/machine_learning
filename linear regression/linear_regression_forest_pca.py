import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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


X_train, X_test, y_train, y_test= train_test_split(data,targets,test_size=0.25,random_state=0)

pca = PCA(n_components=1)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train_pca,y_train)

# Make predictions using the testing set
forest_y_pred = regr.predict(X_test_pca)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(X_test_pca, forest_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, forest_y_pred))
print (regr.score(X_test_pca, y_test))
print (regr.score(X_train_pca, y_train))

plt.scatter(X_train_pca, y_train, color='black')

plt.plot(X_test_pca, forest_y_pred, color='red')

plt.xlabel("Component")
plt.ylabel(headers[12])
plt.legend(loc='upper right')
plt.show()
