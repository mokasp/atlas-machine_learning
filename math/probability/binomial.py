#!/usr/bin/env python3
class Binomial():

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = n
            if p > 1 or p < 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            first = []
            second = []
            for i in range(10):
                for j in range(10):
                    if i * j == data[0]:
                        first.append(i)
                        second.append(j)
            self.n = first[0] * 10
            self.p = sum(data) / (self.n * 10)