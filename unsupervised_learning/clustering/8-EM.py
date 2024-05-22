#!/usr/bin/env python3
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    pi, m, S = initialize(X, k)
    for i in range(iterations):
        if i > 0:
            last_l = l
        g, l = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose == True:
            print(f'Log Likelihood after {i} iterations: {np.round(l, 5)}')
        if i > 0 and l - last_l <= tol:
            print(f'Log Likelihood after {i} iterations: {np.round(l, 5)}')
            return pi, m, S, g, l
        pi, m, S = maximization(X, g)
    return pi, m, S, g, l