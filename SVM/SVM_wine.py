import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# import some data to play with
wine = datasets.load_wine()
X = wine.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
Y = wine.target

def my_kernel(X, Y):
    M = np.array([[4, 0], [0, 2.0]])
    return np.dot(np.dot(X, M), Y.T)


h = .02  # step size in the mesh


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.25, random_state=0)

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

print(clf.score(X_train,Y_train))
print(clf.score(X_test,Y_test))


# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('User defined kernel')
plt.axis('tight')
plt.xticks(())
plt.yticks(())
plt.show()

