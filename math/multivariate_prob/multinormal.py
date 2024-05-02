#!/usr/bin/env python3
import numpy as np


class MultiNormal():

    def __init__(self, data):
        n = data.shape[1]
        d = data.shape[0]

        dat = data.T

        self.mean = np.mean(data, axis=1, keepdims=True)

        cov = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                s = 0
                for h in range(n):
                    s += (dat[h][i] - self.mean[i]) * (dat[h][j] - self.mean[j])
                    cov[i][j] = s / (n -1)
                    cov[j][i] = round(cov[i][j], 8)

        self.cov = cov

    def pdf(self, x):
        d = len(self.mean)

        coeff = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(self.cov)))

        expo = (-1/2) * ((x - self.mean).T @ np.linalg.inv(self.cov) @ (x - self.mean))

        return float(coeff * np.exp(expo))
