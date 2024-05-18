#!/usr/bin/env python3
import numpy as np

def get_centroid(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
    return np.argmin(distances, axis=-1)

def kmeans(X, k, iterations=1000):
    if type(X) != type(np.array([])) or len(X.shape) < 2 or type(X[0][0]) == type(np.array([])) :
        return None
    
    if type(k) != int or k <= 0:
        return None

    d = X.shape[1]

    min_v = np.min(X, axis=0)
    max_v = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_v, high=max_v, size=(k, d))
    for x in range(iterations):
        labels = get_centroid(X, centroids)

        new_centroids = np.zeros((k, d))

        for i in range(k):
            if np.sum(labels == i) <= 0:
                new_centroids[i] = np.random.uniform(low=min_v, high=max_v, size=d)
            else:
                new_centroids[i] = np.mean(X[labels == i], axis=0)

        if np.allclose(new_centroids, centroids):
            labels = get_centroid(X, centroids)
            return centroids, labels
        
        centroids = new_centroids
        
    labels = get_centroid(X, centroids)

    return centroids, labels