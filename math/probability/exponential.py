#!/usr/bin/env python3
""" module containing class that represents an exponential distribution"""


class Exponential():
    """ class that represents a exponential distribution

        Parameters:
            data (list): data used to estimate distribution
            lambtha (int): expected number of occurences in a given time
                            period

        Methods:
            pdf(x): calculates the probability density function of exponential
            cdf(x): calculates the cumulative distribution function of
                    exponential distribution
            """

    def __init__(self, data=None, lambtha=1):
        """ intializes instance of an exponential distribution

        Parameters:
            data (list): data used to estimate distribution
            lambtha (int): expected number of occurences in a given time
                        period
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """ calculates the probability density function of
            exponential distribution

            Parameters:
                x (int): time period
            """
        if x < 0:
            return 0
        e = 2.7182818285
        lam = self.lambtha
        return lam * (e ** (-(lam) * x))

    def cdf(self, x):
        """ calculates the cumulative distribution function of
            exponential distribution

            Parameters:
                x (int): time period
            """
        if x < 0:
            return 0
        e = 2.7182818285
        lam = self.lambtha
        return 1 - e ** (-(lam) * x)
