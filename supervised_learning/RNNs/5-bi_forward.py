#!/usr/bin/env python3
import numpy as np
""" module containing a representation of a bidirectional RNN cell """


class BidirectionalCell():
    """ representation of a Bidirectional RNN cell """

    def __init__(self, i, h, o):
        """ initialize """
        self.Whf = np.random.randn(h + i, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(h + i, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(h + h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward pass """
        h = np.concatenate((h_prev, x_t), axis=1)
        hf_t = np.tanh(np.dot(h, self.Whf) + self.bhf)

        return hf_t

    def softmax(self, x):
        """ softmax activation """
        z = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(z)
        z = exp / np.sum(exp, axis=-1, keepdims=True)
        return z
