#!/usr/bin/env python3
import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a Gaussian Mixture Model.

        Parameters
        ----------
        X : numpy.ndarray
            Data set with shape (n, d).
        g : numpy.ndarray
            Posterior probabilities for each data point in each cluster with shape (k, n).

        Returns
        -------
        pi : numpy.ndarray or None
            Updated priors for each cluster with shape (k,).
            Returns None on failure.
        m : numpy.ndarray or None
            Updated centroid means for each cluster with shape (k, d).
            Returns None on failure.
        S : numpy.ndarray or None
            Updated covariance matrices for each cluster with shape (k, d, d).
            Returns None on failure.
    """
    print(g)
    # get shapes
    n, d = X.shape
    k, _ = g.shape

    # update priors
    pi = (1 / n) * np.sum(g, axis=1)

    # calc numerator
    num = np.dot(g, X)
    # calc denom
    denom = np.sum(g, axis=1)
    # normalize to get new means
    m = (num.T / denom).T

    # create empty matrix to store covariance matrix
    cov = np.zeros((k, d, d))
    # fur each cluster
    for i in range(k):
        # center data around mean
        t = X - m[i]
        # scale data by weight
        h = g[i] * t.T
        # calc weighted sum of squared distances of each dp from the mean
        j = np.dot(h, t)
        # adjust fur varying sizes of components
        cov[i] = j / np.sum(g[i])
    return pi, m, cov