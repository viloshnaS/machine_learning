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

feature=4

X=wine.data[: , feature]
y=wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0,
                                                    stratify=y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)


max_pt = f1.max() * regressor.coef_[0] + regressor.intercept_

print (regressor.score(X_test, y_test))
print (regressor.score(X_train, y_train))