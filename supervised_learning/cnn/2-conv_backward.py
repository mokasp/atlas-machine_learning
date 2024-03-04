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
        pad_h = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                     mode='constant', constant_values=0)
    dW = np.zeros_like(W)
    dX = np.zeros(A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for sample in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for i in range(c_new):
                    dX[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, :] += dZ[sample, row, col, i] * W[:, :, :, i]

                    dW[:, :, :, i] += A_prev[sample, row*sh:row*sh+kh, col*sw:col*sw+kw, :] * dZ[sample, row, col, i]
    if padding == 'same':
        dX = dX[:, pad_h:dX.shape[1]-pad_h, pad_w:dX.shape[2]-pad_w, :]

    return dX, dW, db