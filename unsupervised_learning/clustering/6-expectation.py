#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    d, _ = X.shape
    k, _= m.shape
    nums = np.zeros(shape=(k, d))
    e_step = np.zeros(shape=(k, d))
    for i in range(len(m)):
        nums[i] = pdf(X, m[i], S[i])
        e_step[i] = np.dot(pi[i], nums[i])
    denom = np.sum(e_step, axis=0)
    e_step /= denom
    return e_step, np.sum(np.log(denom))
    
