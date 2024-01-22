#!/usr/bin/env python3
import numpy as np

class Neuron():
    """class neuron"""

    def __init__(self, nx):
        """ initialize """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0