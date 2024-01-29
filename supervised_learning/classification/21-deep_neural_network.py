#!/usr/bin/env python3
""" module containing class NeuralNetwork """
import numpy as np


class DeepNeuralNetwork():
    """class representing a Deep Neural Network with multiple hidden layer that
        performs binary classification

        Instance Attributes:
            L (): number of layers
            cache (): all intermediary values
            weights (dict): all weights and biases
            """

    def __init__(self, nx,  layers):
        """ initialize Deep Neural Network

            Parameters:
                nx (int) - nmber of input features
                layers (int) - number of nodes in the hidden layer """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        layers.insert(0, nx)
        self.__cache = {}
        self.__weights = {}
        self.__weights = {}
        for l in range (1, self.__L + 1):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2.0 / (layers[l- 1]))
            self.__weights["b" + str(l)] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """ getter for [] """
        return self.__L

    @property
    def cache(self):
        """ getter for [] """
        return self.__cache

    @property
    def weights(self):
        """ getter for [] """
        return self.__weights

    def sigmoid(self, z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """ one forward pass of neuron """
        self.__cache["A0"] = X
        A = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            z = np.dot(W, A) + b[0]
            A = self.sigmoid(z)
            self.__cache["A" + str(l)] = A
        return A, self.__cache
    
    def cost(self, Y, A):
        """ [] """
        inner1 = np.multiply(np.log(1.0000001 - A), (1 -Y))
        inner2 = np.multiply(np.log(A), Y) + inner1
        summa = np.sum(inner2)
        cel = (-(1 / Y.shape[1]) / A.shape[0]) * summa
        return cel
 
    def evaluate(self, X, Y):
        """ evaluate networks predictions """
        A, hidden = self.forward_prop(X)
        res = np.where(A >= 0.5, 1, 0)
        return res, self.cost(Y, A)
    
    def sig_back(self, dA):
        return dA * (1 - dA)
    
    def get_z(self, W, b, X):
        return W.dot(X) + b

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ [] """
        N = Y.shape[1]
        leng = len(cache)

        for i in reversed(range(1, leng)):
            if i - 1 > 0 and i < leng:
                A0 = cache["A0"]
                A2 = cache["A" + str(i)]
                W2 = self.__weights["W" + str(i)]
                b2 = self.__weights["b" + str(i)]
                A1 = cache["A" + str(i - 1)]
                W1 = self.__weights["W" + str(i - 1)]
                b1 = self.__weights["b" + str(i - 1)]
                
                dz2 = (cache["A" + str(i)] - Y)

                dW2 = (1 / N) * np.dot(dz2, A1.T)
 
                db2 = (1 / N) * np.sum(dz2, keepdims=True)
                dg1 = cache["A" + str(i - 1)]

                dg = dg1 * (1 - cache["A" + str(i - 1)])
                dz11 = self.__weights["W" + str(i)].T

                dz12 = dz2

                dz1 = (np.dot(dz11, dz12)) * dg

                dw11 = np.dot(dz1, cache["A0"].T[:, :self.__weights["W" + str(i - 1)].shape[1]])

                dw1 = (1 / N) * dw11
                db1 = (1 / N) * np.sum(dz1, axis=1, keepdims=True)





                temp = alpha * dw1
                temp2 = alpha * db1
                if i == leng - 1:
                    self.__weights["W" + str(i)] -= alpha * dW2
                    self.__weights["b" + str(i)] -= alpha * db2
                self.__weights["W" + str(i - 1)] = self.__weights["W" + str(i - 1)] - temp
                self.__weights["b" + str(i - 1)] = self.__weights["b" + str(i - 1)] - temp2

