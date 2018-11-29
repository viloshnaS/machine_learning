import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
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
feature = 6

X = data

y = targets

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0)

# Fit regression model
svr_lin = SVR(kernel='linear')
svr_rbf = SVR(kernel='rbf', gamma=0.1)
svr_poly = SVR(kernel='poly')

print("x")
y_poly=svr_poly.fit(X_train, y_train).predict(X_train)

print("y")
print("Polynomial train error: ", mean_squared_error(y_train, y_poly)," test error: ", mean_squared_error(y_test, svr_rbf.predict(X_test)))


plt.scatter(X_train[:, feature], y_train, color='darkorange', label='data')
#plt.scatter(X_train[:, feature], y_lin, color='c', label='Linear model')
plt.scatter(X_train[:, feature], y_poly, color='red', label='Polynomial model')

print('accuracy')

print('Poly', svr_poly.score(X_test, y_test))
print('Poly', svr_poly.score(X_train, y_train))

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
