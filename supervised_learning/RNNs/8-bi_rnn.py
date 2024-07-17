#!/usr/bin/env python3
import numpy as np
""" module containing function that performs full forward propagation
    with a bidirectional RNN """


def bi_rnn(rnn_cell, X, h_0, h_t):
    """ function that performs full forward propagation of a
    bidirectional RNN """

    h_for = []
    h_back = []
    t_steps = X.shape[0]
    for i in range(t_steps):
        h_0 = rnn_cell.forward(h_0, X[i])
        h_for.append(h_0)
        h_t = rnn_cell.backward(h_t, X[t_steps - (i + 1)])
        h_back.append(h_t)
    h_for = np.array(h_for)
    h_back = np.array(h_back[::-1])

    H = np.concatenate((h_for, h_back), axis=2)
    Y = rnn_cell.output(H)
    return H, np.array(Y)
