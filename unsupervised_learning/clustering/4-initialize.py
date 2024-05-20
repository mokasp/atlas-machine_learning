#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    n, d = X.shape
    m, _ = kmeans(X, k)
    s = np.identity(d)
    S = np.broadcast_to(s, (k, d, d))
    prior = (100 / k) * 0.01
    pi = np.full((k,), prior)
    return pi, m, S