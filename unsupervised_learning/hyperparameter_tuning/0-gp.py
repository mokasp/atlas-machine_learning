#!/usr/bin/env python3
""" module containing class that represents a noiseless 1D gaussian
    process """
import numpy as np


class GaussianProcess():
    """ represents a noiseless 1d gaussian process

        Attributes
        ----------
        X : numpy.ndarray
            Inputs, shape (t, 1).
        Y : numpy.ndarray
            Outputs, shape (t, 1).
        l : float
            Length parameter for the kernel.
        sigma_f : float
            Standard deviation given to the output of the black-box
            function.
        K : numpy.ndarray
            Current covariance kernel matrix for the Gaussian process.

        Methods
        -------
        kernel : input mat1, input mat2
            function that calculates the covariance kernel matrix between
            two matrices using the Radial Basis Function (RBF) kernel
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Create the class GaussianProcess that represents a noiseless 1D
            Gaussian process. This class initializes a noiseless 1D Gaussian
            process

            Parameters
            ----------
            X_init : numpy.ndarray
                Inputs already sampled with the black-box function,
                shape (t, 1).
            Y_init : numpy.ndarray
                Outputs of the black-box function for each input in X_init,
                shape (t, 1).
            l : float, optional
                Length parameter for the kernel, default value is 1.
            sigma_f : float, optional
                Standard deviation given to the output of the black-box
                function, default value is 1.

            Attributes
            ----------
            X : numpy.ndarray
                Inputs, shape (t, 1).
            Y : numpy.ndarray
                Outputs, shape (t, 1).
            l : float
                Length parameter for the kernel.
            sigma_f : float
                Standard deviation given to the output of the black-box
                function.
            K : numpy.ndarray
                Current covariance kernel matrix for the Gaussian process.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        X = np.concatenate((self.X, self.X), axis=1)
        Y = np.concatenate((self.X.T, self.X.T), axis=0)
        K = self.sigma_f ** 2 * np.exp(-1 * (((X - Y) ** 2 /
                                              (2 * self.ll ** 2))))
        self.K = K

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices using
            the Radial Basis Function (RBF) kernel.

            Parameters
            ----------
            X1 : numpy.ndarray
                Input matrix 1, shape (m, 1).
            X2 : numpy.ndarray
                Input matrix 2, shape (n, 1).

            Returns
            -------
            K : numpy.ndarray
                Covariance kernel matrix, shape (m, n).
        """
        X = np.concatenate((X1, X1), axis=1)
        Y = np.concatenate((X2.T, X2.T), axis=0)
        K = self.sigma_f ** 2 * np.exp(-1 * (((X - Y) ** 2 / (2
                                                              * self.l ** 2))))
        return K
