#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    if kmax == None:
        kmax = 20
    results, labels, d_vars = [], [], []
    for i in range(kmin, kmax + 1):
        res, lab = kmeans(X, i, iterations)
        results.append((res, lab))
        var = variance(X, res)
        small_var = variance(X, results[0][0])
        d_vars.append(small_var - var)
    return results, d_vars