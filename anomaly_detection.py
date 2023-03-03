from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from numpy import quantile, where, random, dot
import matplotlib.pyplot as plt

random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.002, nu=0.1)

SVMmodelOne.fit(x)
scores = SVMmodelOne.score_samples(x)

thresh = quantile(scores, 0.1)
print(thresh)
index = where(scores<=thresh)
values = x[index]

supvectors=SVMmodelOne.support_vectors_
decision_function = SVMmodelOne.decision_function(x)
#decision_function = dot(x, SVMmodelOne.coef_[0]) + SVMmodelOne.intercept_[0]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')

plt.scatter(
    supvectors[:, 0],
    supvectors[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    SVMmodelOne,
    x,
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)

plt.tight_layout()
plt.axis('equal')
plt.show()
