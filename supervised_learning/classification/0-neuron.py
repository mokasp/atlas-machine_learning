#!/usr/bin/env python3
import numpy as np

class Neuron():
    """class neuron"""

    def __init__(self, nx):
        """ initialize """
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0