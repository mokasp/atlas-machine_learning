#!/usr/bin/env python3
""" module containing class that represents a poisson distribution"""


class Poisson():
    """ class that represents a poisson distribution

        Parameters:
            data (list): data used to estimate distribution
            lambtha (int): expected number of occurences in a given time
                            period

        Methods:
            pmf(k): calculates the probability mass function of poisson
            cdf(k): calculates the cumulative distribution function of
                    poisson distribution
            """

    def __init__(self, data=None, lambtha=1):
        """ intializes instance of a poisson distribution

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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ calculates the probability mass function of poisson distribution

            Parameters:
                k (int): number of successes
            """
        if k < 0:
            return 0
        e = 2.7182818285
        mu = self.lambtha
        k = int(k)
        fact_list = [*range(1, k + 1, 1)]
        denom = 1
        for item in fact_list:
            denom = denom * item
        return ((e ** -(mu)) * (mu ** k)) / denom

    def cdf(self, k):
        """ calculates the cumulative distribution function of poisson dist

            Parameters:
                k (int): number of successes
        """
        if k < 0:
            return 0
        e = 2.7182818285
        mu = self.lambtha
        k = int(k)
        i = [*range(0, k + 1, 1)]
        summ = 0
        for elem in i:
            fact_list = [*range(1, elem + 1, 1)]
            denom = 1
            for item in fact_list:
                denom = denom * item
            summ += ((e ** -(mu)) * (mu ** elem)) / denom
        return summ
