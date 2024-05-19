#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    if kmax == None:
        kmax = 40
    results, labels, d_vars = [], [], []
    patience = 0
    for i in range(kmin, kmax + 1):
        res, lab = kmeans(X, i, iterations)
        results.append((res, lab))
        var = variance(X, res)
        small_var = variance(X, results[0][0])
        d_vars.append(small_var - var)
        if i > 1 and d_vars[i - 1] > d_vars[i - 2]:
            if d_vars[i - 1] - d_vars[i - 2] < 1:
                patience += 1
            if patience == 2:
                break
    return results, d_vars