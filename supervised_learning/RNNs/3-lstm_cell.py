#!/usr/bin/env python3
import numpy as np
""" module containing a representation of an LSTM cell """


class LSTMCell():
    """ representation of a LSTM cell """
    

    def __init__(self, i, h, o):
        """ initialize """
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))
    
    def forward(self, h_prev, c_prev, x_t):
        """ forward pass """
        temp = np.ndarray([])
        return temp, temp, temp
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """ softmax activation """
        z = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(z)
        z = exp / np.sum(exp, axis=-1, keepdims=True)
        return z