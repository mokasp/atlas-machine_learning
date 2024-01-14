#!/usr/bin/env python3
class Poisson():

    def __init__(self, data=None, lambtha=1):
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
        e = 2.7183
        mu = self.lambtha
        k = int(k)
        fact_list = [*range(1, k + 1, 1)]
        denom = 1
        for item in fact_list:
            denom = denom * item
        return ((e ** -(mu)) * (mu ** k)) / denom

