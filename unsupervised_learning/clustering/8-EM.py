#!/usr/bin/env python3
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    message = 'Log Likelihood after {} iterations: {}'
    pi, m, S = initialize(X, k)
    for i in range(iterations):
        if i > 0:
            prev_likelihood = likelihood
        g, likelihood = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose is True:
            print(message.format(i, round(likelihood, 5)))
        if i > 0 and abs(likelihood - prev_likelihood) <= tol:
            if verbose:
                print(message.format(i, round(likelihood, 5)))
            return pi, m, S, g, likelihood
        pi, m, S = maximization(X, g)
    if verbose:
        print(message.format(i, round(likelihood, 5)))
    return pi, m, S, g, likelihood
