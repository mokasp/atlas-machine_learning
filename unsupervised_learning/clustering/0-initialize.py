#!/usr/bin/env python3
import numpy as np


def initialize(X, k):

    if type(X) != type(np.array([])) or len(X.shape) < 2 or type(X[0][0]) == type(np.array([])) :
        return None
    
    if type(k) != int or k <= 0:
        return None

    d = X.shape[1]

    min_v = np.min(X, axis=0)
    max_v = np.max(X, axis=0)

    return np.random.uniform(low=min_v, high=max_v, size=(k, d))