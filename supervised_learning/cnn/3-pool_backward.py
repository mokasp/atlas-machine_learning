#!/usr/bin/env python3
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out = np.zeros(A_prev.shape)
    for sample in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for i in range(c_new):
                    if mode == 'max':
                        cur = A_prev[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, i]
                        maxed = (cur == np.max(cur))
                        out[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, i] += dA[sample, row, col, i] * maxed
                    else:
                        da = dA[sample, row, col, i]
                        avg = da / (kh * kw)
                        out[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, i] += np.ones(kernel_shape) * avg
    return out