#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):

    standardized = X 

    cov = np.cov(standardized, rowvar=False)

    o_values, o_vectors = np.linalg.eig(cov)
    values = np.copy(o_values)
    vectors = np.copy(o_vectors)

    sort = np.argsort(values)[::-1]
    s_values = values[sort]
    s_vectors = vectors[:, sort]

    v_ex = []
    for i in s_values:
        v_ex.append((i/sum(s_values)))

    total = np.sum(s_values)

    test = np.cumsum(s_values) / total

    
    r_cumsum = np.argmax(test >= var) + 2


    selected = s_vectors[:, :r_cumsum]

    return selected
