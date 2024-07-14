#!/usr/bin/env python3
import numpy as np
""" module containing function that performs full forward propagation
    with a deep RNN """


def deep_rnn(rnn_cells, X, h_0):
    """ function that performs full forward propagation of adeep RNN """
    H = []
    Y = []
    H.append(h_0)
    for i in range(X.shape[0]):
        t_H = []
        x = X[i]
        for cell in range(len(rnn_cells)):
            h_prev, y = rnn_cells[cell].forward(H[i][cell], x)
            t_H.append(h_prev)
            x = h_prev
        Y.append(y)
        H.append(t_H)
    
    return np.array(H), np.array(Y) 


    # H = []
    # h_now = []
    # for cell in range(len(rnn_cells)):
    #     h_prev = h_0[cell]
    #     h_now.append(h_prev)
    #     Y = []
    #     for i in range(X.shape[0]):
    #         h_prev, y = rnn_cells[cell].forward(h_prev, X[i])
    #         h_now.append(h_prev)
    #         Y.append(y)
    #     X = np.array(Y)
    #     H.append(h_now)

    # return np.array(H), np.array(Y)