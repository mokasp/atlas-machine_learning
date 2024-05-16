#!/usr/bin/env python3
import numpy as np

def pca(X, ndim):

    standardized = (X - np.mean(X, axis=0)) 

    cov = np.cov(standardized, rowvar=False)

    o_values, o_vectors = np.linalg.eigh(cov)
    values = np.copy(o_values)
    vectors = np.copy(o_vectors)

    sort = np.argsort(values)[::-1]
    s_vectors = vectors[:, sort]


    selected = s_vectors[:, :ndim]

    selected[:, 1] *= -1

    transformed = np.matmul(standardized, selected)

    return transformed
