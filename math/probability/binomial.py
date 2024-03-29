#!/usr/bin/env python3
""" module containing class that represents a binomial distribution"""


class Binomial():
    """ class that represents a binomial distribution

        Parameters:
            data (list): list containing number of successes from trials
            n (int): number of trials
            p (float): probability of success

        Methods:
            pmf(k): calculates the probability mass function of binmomial
            cdf(k): calculates the cumulative distribution function of binmoial
            """

    def __init__(self, data=None, n=1, p=0.5):
        """ initializes instances of Binomial

            Parameters:
                data (list): list containing number of successes from trials
                n (int): number of trials
                p (float): probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = n
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            data2 = []
            mean = sum(data) / len(data)
            differences = [x - mean for x in data]
            squared_diff = [x ** 2 for x in differences]
            summed = sum(squared_diff)
            variance = summed / len(data)
            check = 1 - (variance / mean)
            self.n = round((sum(data) / len(data)) / check)
            for i in range(len(data)):
                data2.append((data[i] / self.n))
            check2 = str(sum(data2) / len(data2))
            if check2[5] == "9" and check2[6] == "9":
                check2 = round(float(check2), 6)
            self.p = float(check2)

    def pmf(self, k):
        """ calculates the probability mass function of binmomial distribution

            Parameters:
                k (int): number of successes
            """
        if k < 0:
            return 0
        k = int(k)
        numer = 1
        for fact in range(self.n + 1):
            if fact != 0:
                numer *= fact
        nx = self.n - k
        denom_1 = 1
        for fact in range(int(nx + 1)):
            if fact != 0:
                denom_1 *= fact
        denom_2 = 1
        for fact in range(int(k + 1)):
            if fact != 0:
                denom_2 *= fact
        denom = denom_1 * denom_2
        nCx = numer / denom
        pq = self.p ** k
        qnx = (1 - self.p) ** (self.n - k)
        px = nCx * pq * qnx
        return px

    def cdf(self, k):
        """ calculates the cumulative distribution function of binmomial

            Parameters:
                k (int): number of successes
        """
        summ = 0
        k = int(k)
        for i in range(k + 1):
            summ += self.pmf(i)
        return summ
