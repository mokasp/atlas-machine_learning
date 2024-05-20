#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    if type(X) != type(np.array([])) or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k < 1:
        return None, None, None
    n, d = X.shape
    m, _ = kmeans(X, k)
    s = np.identity(d)
    S = np.broadcast_to(s, (k, d, d))
    prior = (100 / k) * 0.01
    pi = np.full((k,), prior)
    return pi, m, S