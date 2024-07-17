#!/usr/bin/env python3
import numpy as np
""" module containing a representation of a Gated Recurrent Unit cell """


class GRUCell():
    """ representation of a Gated Recurrent Unit cell """

    def __init__(self, i, h, o):
        """ initialize """
        self.Wz = np.random.randn(h + i, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(h + i, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward pass """
        h = np.concatenate((h_prev, x_t), axis=1)

        z_t = self.sigmoid(np.dot(h, self.Wz) + self.bz)
        r_t = self.sigmoid(np.dot(h, self.Wr) + self.br)

        h_tt = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat_t = np.tanh(np.dot(h_tt, self.Wh) + self.bh)
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t
        y = self.softmax(np.dot(h_t, self.Wy) + self.by)

        return h_t, y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ softmax activation """
        z = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(z)
        z = exp / np.sum(exp, axis=-1, keepdims=True)
        return z
