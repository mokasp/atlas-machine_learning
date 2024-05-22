#!/usr/bin/env python3
import sklearn.cluster


def kmeans(X, k):
    k_means = sklearn.cluster.KMeans(k).fit(X)
    return k_means.cluster_centers_, k_means.labels_