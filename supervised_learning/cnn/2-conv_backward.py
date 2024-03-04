#!/usr/bin/env python3
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h = int(((h_prev * sh) - h_prev + kh - sh) / 2)
        pad_w = int(((w_prev * sw) - w_prev + kh - sw) / 2)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                     mode='constant')
    dW = np.zeros(W.shape)
    dX = np.zeros(A_prev.shape)
    db = np.zeros((1, 1, 1, c_new))
    for sample in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for i in range(c_new):
                    dX[:, row*sh:row*sh+kh, col*sw:col*sw+kw, :] += dZ[sample, row, col, i] * W[:, :, :, i]

                    dW[:, :, :, i] += A_prev[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, :] * dZ[sample, row, col, i]

                    db[:, :, :, i] += dZ[sample, row, col, i]

    return dX, dW, db