#!/usr/bin/env python3
import numpy as np
""" module containing function that performs full forward propagation
    with an RNN """


def rnn(rnn_cell, X, h_0):
    """ function that performs full forward propagation of an RNN """
    h_prev = h_0
    H = []
    Y = []
    H.append(h_prev)
    for i in range(X.shape[0]):
        h_prev, y = rnn_cell.forward(h_prev, X[i])
        H.append(h_prev)
        Y.append(y)

    return np.array(H), np.array(Y)
