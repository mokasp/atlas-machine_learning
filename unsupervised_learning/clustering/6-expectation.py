#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    # check validity of X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None

    # check validity of m
    if not isinstance(m, type(np.array([]))) or len(m.shape) < 2 or isinstance(m[0][0], type(np.array([]))):
        return None, None

    # check validity of pi
    if not isinstance(pi, type(np.array([]))) or len(
            pi.shape) > 1:
        return None, None

    # check validity of S
    if not isinstance(S, type(np.array([]))) or \
        len(S.shape) < 3 or \
            S.shape[1] != S.shape[2] or \
                S.shape[0] != pi.shape[0]:
        return None, None

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
    
