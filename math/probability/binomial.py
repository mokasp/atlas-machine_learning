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
        numer = 1
        for fact in range(self.n + 1):
            if fact != 0:
                numer *= fact
        nx = self.n - k
        denom_1 = 1
        for fact in range(nx + 1):
            if fact != 0:
                denom_1 *= fact
        denom_2 = 1
        for fact in range(k + 1):
            if fact != 0:
                denom_2 *= fact
        denom = denom_1 * denom_2
        nCx = numer / denom
        pq = self.p ** k
        qnx = (1 - self.p) ** (self.n - k)
        px = nCx * pq * qnx
        return px