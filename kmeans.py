import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import math
from matplotlib.image import imread

def euclidian_distance(centroid, data) -> list:
    arr = np.empty(len(data))
    for i in range(len(data)):
        point = data[i]
        dist = 0
        for j in range(len(centroid)):
            dist += np.square(point[j] - centroid[j])
        arr[i] = np.sqrt(dist)
    return arr

def kmean_iteration(centroids, data):
    dist_mat = np.zeros((len(data), len(centroids)))

    for n in range(len(centroids)):
        distance = euclidian_distance(centroids[n], data)
        dist_mat[:, n] = distance

    clusters = []
    for row in dist_mat:
        clusters.append(np.argmin(row))
    
    return clusters

def kmean(k, dataset):
    centers = k


    #TODO make a method that determines the  optimal number of clusters. Try elbow curve.
    centroids = random.sample(list(dataset), centers) 
    new_centroids = np.copy(centroids)

    centroid_moving = True
    current_iteration = 0
    maximum_iteration = 25

    while centroid_moving:
        clusters = kmean_iteration(centroids, dataset)

        for i in range(len(centroids)):
            proxy_mat = np.c_[clusters, dataset]
            new_centroids[i] = np.mean(proxy_mat[proxy_mat[:, 0] == i, 1:], axis=0)
        
        current_iteration += 1

        if current_iteration > maximum_iteration:
            print("Maximum precision reached")
            centroid_moving = False
            
        elif np.allclose(centroids, new_centroids):
            print("Centroids not moving in iterations", current_iteration)
            centroid_moving = False

        centroids = np.copy(new_centroids)
    
    return centroids, clusters

def get_sum_square_distance(centroids, clusters, dataset):
    sum = 0
    proxy_mat = np.c_[clusters, dataset]

    for i in range(len(centroids)):
        points = proxy_mat[proxy_mat[:, 0] == i, 1:]
        for point in points:
            sum += math.dist(point, centroids[i])
    
    print(math.pow(sum, 2))
    return math.pow(sum, 2)

def elbow_curve(dataset):
    possible_k = [(X+1) for X in range(10)]
    sum_of_squared_distance = np.empty(len(possible_k))
    for i in range(len(possible_k)):
        centroids, clusters = kmean(possible_k[i], dataset)
        sum_of_squared_distance[i] = get_sum_square_distance(centroids, clusters, dataset)
    
    plt.figure()
    plt.plot(possible_k, sum_of_squared_distance)
    plt.show()

def compress_image(image: np.array, nb_of_colours: int):
    original_shape = image.shape
    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    print(image.shape)
    print(image)
    #centroids, clusters = kmean(nb_of_colours, image)
    clusters = np.array(clusters)
    print(clusters)
    new_image = clusters.reshape(original_shape)
    plt.imshow(new_image)
    plt.show()

#You can plot true_labels for comparaison with the result of the algorithm.
X_train, true_labels = make_blobs(n_samples=1000, centers=3, random_state=40)
X_train = StandardScaler().fit_transform(X_train)

k = 3;

#elbow_curve(X_train)

image = imread("kmeans/fish.jpg")
compress_image(image, 30)

centroids, clusters = kmean(k, X_train);

x = [X[0] for X in X_train]
x.extend(centroids[:, 0])
y = [X[1] for X in X_train]
y.extend(centroids[:, 1])

clusters.extend([9] * k)

sns.scatterplot(x=x,
                y=y,
                legend=None,
                hue=clusters,
                palette="deep"
                )

plt.xlabel("x")
plt.ylabel("y")
#plt.show()
