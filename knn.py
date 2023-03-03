# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.model_selection import train_test_split
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

iris=load_iris()

X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

fig, axs = plt.subplots(1, 3, figsize=(9, 3))

knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train, y_train)
Ypred=knn1.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, Ypred, ax=axs[0])



Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

pca = decomposition.PCA(n_components=2)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,y,test_size=0.3)

knn2=KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_train_pca, y_train_pca)
Ypred_pca=knn2.predict(X_test_pca)
ConfusionMatrixDisplay.from_predictions(y_test_pca, Ypred_pca, ax=axs[1])



X=iris.data[:, :2]
y=iris.target

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X,y,test_size=0.3)

knn3=KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train_2d, y_train_2d)
Ypred=knn3.predict(X_test_2d)
ConfusionMatrixDisplay.from_predictions(y_test_2d, Ypred, ax=axs[2])

axs[0].title.set_text('Full Dataset')
axs[1].title.set_text('PCA with 2 components')
axs[2].title.set_text('2 first columns')

fig.tight_layout()
plt.show()
