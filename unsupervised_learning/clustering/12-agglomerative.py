#!/usr/bin/env python3
""" module containing function that performs K-means clustering on a dataset
    using sklearn. """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ function that performs K-means clustering on a dataset using sklearn.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset with shape (n, d).
        k : int
            Number of clusters.

        Returns
        -------
        C : numpy.ndarray
            Centroid means for each cluster with shape (k, d).
        clss : numpy.ndarray
            Index of the cluster in C that each data point belongs to
            with shape (n,).
    """
    z = scipy.cluster.hierarchy.linkage(X, 'ward')
    den = scipy.cluster.hierarchy.dendrogram(z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(z, t=dist, criterion='distance')
    plt.show()
    return clss
