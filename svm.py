from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()

X=iris.data[iris.target!=2, :2]
y=iris.target[iris.target!=2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

classifier = SVMmodel=SVC(kernel='linear', C=10).fit(X_train,y_train)
SVMmodel.get_params()
print(SVMmodel.score(X_test,y_test))

#decision_function = np.dot(X, classifier.coef_[0]) + classifier.intercept_[0]
decision_function = classifier.decision_function(X_train)

supvectors=SVMmodel.support_vectors_

sns.scatterplot(
    x=X_train[:, 0],
    y=X_train[:, 1],
    hue=y_train
)

ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    classifier,
    X_train,
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
plt.scatter(
    supvectors[:, 0],
    supvectors[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)


plt.xlabel("x")
plt.ylabel("y")
plt.show()
