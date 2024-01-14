#!/usr/bin/env python3
class Exponential():

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
            self.lambtha = 1 / (sum(data) / len(data))
    
    def pdf(self, x):
        if x < 0:
            return 0
        e = 2.7182818285
        lam = self.lambtha
        return lam * (e ** (-(lam) * x))
    
    def cdf(self, x):
        if x < 0:
            return 0
        e = 2.7182818285
        lam = self.lambtha
        return 1 - e ** (-(lam) * x)