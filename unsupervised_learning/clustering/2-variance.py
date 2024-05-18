#!/usr/bin/env python3
import numpy as np

def get_centroid(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
    return np.argmin(distances, axis=-1), distances ** 2

def variance(X, C):

    # get labels with which centroids each point belongs to, along with the squared distances
    labels, distances = get_centroid(X, C)

    # create empty matrix to store the squared distance from each datapoint to its closest centroid in the corresponding index position
    masked = np.zeros_like(distances)

    # store the distance to the closest centroid in the index that corresponds to that centroid. leave every other position in that row blank
    masked[np.arange(len(X)), labels] = distances[np.arange(len(X)), labels]

    # add all of the squared distances together to get total intracluster variance
    total_variance = np.sum(masked)

    return total_variance