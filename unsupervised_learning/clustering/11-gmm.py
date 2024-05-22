#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):
    g_mm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = g_mm.weights_
    m = g_mm.means_
    S = g_mm.covariances_
    clss = g_mm.predict(X)
    bic = g_mm.bic(X)
    return pi, m, S, clss, bic