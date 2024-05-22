#!/usr/bin/env python3
import numpy as np


def maximization(X, g):
    n, d = X.shape
    k, _ = g.shape
    pi = (1 / n) * np.sum(g, axis=1)
    num = np.dot(g, X)
    denom = np.sum(g, axis=1)
    m = (num.T / denom).T
    cov = np.zeros((k, d, d))
    for i in range(k):
        t = X - m[i]
        h = g[i] * t.T
        j = np.dot(h, t)
        cov[i] = j / np.sum(g[i])
    return pi, m, cov