#!/usr/bin/env python3
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    bic_s = []
    l_s = []
    for i in range(kmin, kmax):

        # calc EM for each cluster sie
        pi, m, S, g, l = expectation_maximization(X, i, iterations, tol, verbose)

        # get shapes
        k = pi.shape[0]
        _, d = m.shape
        n, _ = X.shape

        # calc number of params
        p = (k * d) + (k * d * d) + (k - 2)

        # BIC calc
        bic = p * np.log(n) - 2 * l

        # store best results
        if i == 1:
            best_log = l
            best_res = (pi, m, S)
        
        if l <= best_log:
            best_log = l
            best_k = i
            best_res = (pi, m, S)
        


        # add BIC and likelihood to list
        bic_s.append(round(bic, 8))
        l_s.append(np.round(l, 8))
    
    return best_k, best_res, np.array(l_s), np.array(bic_s)
