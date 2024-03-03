#!/usr/bin/env python3
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    op_h = int((h - kh) / sh) + 1
    op_w = int((w - kw) / sw) + 1
    pooled = np.zeros((m, op_h, op_w, c))
    for row in range(op_h):
        for col in range(op_w):
            cur = A_prev[:, row*sh:row*sh+kh, col*sw:col*sw+kw]
            if mode == 'avg':
                pooled[:, row, col] = np.mean(cur, axis=(1, 2))
            else:
                pooled[:, row, col] = np.max(cur, axis=(1, 2))
    return pooled