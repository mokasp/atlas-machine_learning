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