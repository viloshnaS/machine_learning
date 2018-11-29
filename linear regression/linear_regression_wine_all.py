import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the wine dataset
wine = datasets.load_wine()


X=wine.data
y=wine.target

feature1=5
feature2=6



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)


f1 = X[:, feature1]
f2 = X[:, feature2]


min_pt = f1.min() * regressor.coef_[0] + regressor.intercept_
max_pt = f1.max() * regressor.coef_[0] + regressor.intercept_

print (regressor.score(X_test, y_test))
print (regressor.score(X_train, y_train))
plt.plot([f1.min(), f1.max()], [min_pt, max_pt])
plt.scatter(f1[y == 0], f2[y == 0], c='blue', s=40, label='0')
plt.scatter(f1[y == 1], f2[y == 1], c='red', s=40, label='1')
plt.scatter(f1[y == 2], f2[y == 2], c='yellow', s=40, label='2')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(loc='upper right')
plt.show()