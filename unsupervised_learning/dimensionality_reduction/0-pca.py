#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):

    std = np.std(X, axis=0)

    standardized = X / std

    cov = np.cov(standardized, rowvar=False)

    values, vectors = np.linalg.eig(cov)

    sort = np.argsort(values)[::-1]
    s_values = np.real(values[sort])
    s_vectors = np.real(vectors[:, sort])


    total = np.sum(s_values)

    test = s_values / total

    cumsum = np.cumsum(test)
    r_cumsum = np.argmax(cumsum >= var) + 1


    selected = s_vectors[:, :r_cumsum]

    return selected
