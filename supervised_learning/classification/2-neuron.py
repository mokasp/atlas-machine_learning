#!/usr/bin/env python3
import numpy as np

class Neuron():
    """ class neuron """

    def __init__(self, nx):
        """ initialize [] """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        """ getter for [] """
        return self.__W
    
    @property
    def b(self):
        """ getter for [] """
        return self.__b
    
    @property
    def A(self):
        """ getter for [] """
        return self.__A
    
    def sigmoid(self, z):
        """ """
        return 1 / (1 + np.exp(-z))
    
    def forward_prop(self, X):
        """ """
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A