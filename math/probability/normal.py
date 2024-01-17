#!/usr/bin/env python3
class Normal():

    def __init__(self, data=None, mean=0, stddev=1):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            differences = [x - self.mean for x in data]
            squared_diff = [x ** 2 for x in differences]
            summed = sum(squared_diff)
            variance = summed / len(data)
            self.stddev = variance ** (1/2)

    def z_score(self, x):
        return (x - self.mean) / self.stddev
    
    def x_value(self, z):
        return (self.stddev * z) + self.mean

    def pdf(self, x):
        pi = 3.1415926536
        e = 2.7182818285
        first = (1 / (self.stddev * ((2 * pi) ** (1/2))))
        third = -(((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        second = e ** third
        product = first * second
        return product
    
    def cdf(self, x):
        pi = 3.1415926536
        power = [3, 5, 7, 9]
        denom = [3, 10, 42, 216]
        erf_1 = ((x - self.mean)) / (self.stddev * (2 ** (1 / 2)))
        for i in range(len(power)):
            erf_numer = (((x - self.mean)) / (self.stddev * (2 ** (1 / 2)))) ** power[i]
            erf_2 = erf_numer / denom[i]
            if i == 0 or i == 2:
                erf_1 -= erf_2
            elif i == 1 or i == 3:
                erf_1 += erf_2
        erf_3 = 2 / (pi ** (1/2))
        erf = erf_1 * erf_3
        inner = 1 + erf
        product = (1 / 2) * inner
        return product