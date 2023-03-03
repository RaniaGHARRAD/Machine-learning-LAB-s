import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:

#R = np.cov(X)
R = np.matmul(( 1/X.shape[0]) * X, X.T)

# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

# Calculate the coordinates in new orthonormal basis:

# Calculate the approximation of the original from new basis
#print(Xi1[:,None]) # add second dimention to array and test it


# Check that you got the original

xi1=np.matmul(X.T, u1)
xi2=np.matmul(X.T, u2)

X_new = np.matmul(u1[:, None], xi1[None, :]) + np.matmul(u2[:, None], xi2[None, :])
print(X_new)

#------PART 2 IRIS DATASET-------

iris=load_iris()
iris.feature_names

X=iris.data
y=iris.target

Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
# you can plot the transformed feature space in 3D:
""" axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta') """

with plt.style.context('ggplot'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(3), pca.explained_variance_, alpha=0.5, align='center',
    label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xticks(range(3))
    plt.legend()
    plt.tight_layout()
    plt.show()

plot = plt.scatter(Xpca[:,0], Xpca[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=list(iris.feature_names))
plt.show()
