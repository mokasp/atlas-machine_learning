#!/usr/bin/env python3
import numpy as np


def marginal(x, n, P, Pr):
    temp = []

    for i in range(len(P)):
        n_f = 1
        for j in range(1, n + 1):
            n_f *= j
        x_f = 1
        nx_f = 1
        for j in range(1, x + 1):
            x_f *= j
        for j in range(1, (n - x) + 1):
            nx_f *= j
        coeff = n_f / (x_f * nx_f)
        Lp = coeff * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        ints = (Lp * Pr[i])
        temp.append(ints)
    return sum(temp)