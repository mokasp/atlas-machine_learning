#!/usr/bin/env python3
""" module containing function that updates the weights and biases of a
    neural network using gradient descent with L2 regularization """
import numpy as np



def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ function that updates the weights and biases of a neural network
        using gradient descent with L2 regularization


        PARAMETERS
        ==========
        Y [np.ndarray]: one-hot array of shape (classes, m) that contains the
                        correct labels for the data
                        > classes [int]: number of classes
                        > m [int]: number of data points

        weights [dict]: the weights and biases of the neural network

        cache [dict]: outputs of each layer of the neural network

        alpha [float]: learning rate

        lambtha [?]:  L2 regularization parameter

        L [int]: number of layers of the network


        RETURNS
        =======
        None
    """
    N = Y.shape[1]
    W_cur = weights["W" + str(L)]
    A_cur = cache["A" + str(L)]
    A_prev = cache["A" + str(L - 1)]
    b_cur = weights["b" + str(L)]
    adj = {}

    dz2 = (A_cur - Y)
    dW2 = (1 / N) * np.dot(dz2, A_prev.T)
    db2 = (1 / N) * np.sum(dz2, axis=1, keepdims=True)

    adj["W" + str(L - 1)] = W_cur - alpha * (W_cur * (dW2 + (lambtha / N)))
    adj["b" + str(L - 1)] = b_cur - alpha * (W_cur * (db2 + (lambtha / N)))

    for lay in range(L - 1, 0, -1):
        if lay > 0:
            W_cur = weights["W" + str(lay)]
            W_prev = weights["W" + str(lay + 1)]
            A_cur = cache["A" + str(lay)]
            A_prev = cache["A" + str(lay - 1)]
            b_cur = weights["b" + str(lay)]
            dg = 1 - (np.tanh(A_cur) ** 2) 
            dz1 = (np.dot(W_prev.T, dz2)) * dg
            dw1 = (1 / N) * np.dot(dz1, A_prev.T)
            db1 = (1 / N) * np.sum(dz1, axis=1, keepdims=True)
            dz2 = dz1

            adj["W" + str(lay)] = W_cur - alpha * (W_cur * (dw1 + (lambtha / N)))
            adj["b" + str(lay)] = b_cur - alpha * (W_cur * (db1 + (lambtha / N)))

    weights = adj