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
        """ sigmoid function """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """ one forward pass of neuron """
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A

    def cost(self, Y, A):
        """ cost of neuron """
        inner1 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        inner2 = np.multiply(np.log(A), Y) + inner1
        summa = np.sum(inner2)
        cel = (-1 / A.shape[1]) * summa
        return cel

    def evaluate(self, X, Y):
        """ evaluate neurons predictions """
        res = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        return res, self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ calculates one pass of gradiendt descent """
        N = X.shape[1]

        dldw = (1 / N) * np.dot((A - Y), X.T)
        dldb = (1 / N) * np.sum(A - Y)

        self.__W -= alpha * dldw.reshape(self.__W.shape)
        self.__b -= alpha * dldb
