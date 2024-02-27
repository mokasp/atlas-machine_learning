#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ function that 
        
        PARAMETERS
        ==========
            []: 
            []:
        
        RETURNS
        =======
            []:
    """
    kern1, kern2 = kernel.shape
    m, h, w = images.shape
    sh, sw = stride
    if padding == 'same':
        pad_tb = int(round((kern1 - 1) / 2))
        pad_lr = int(round((kern2 - 1) / 2))
        op_h = int(round(h / sh))
        op_w = int(round(w / sw))
        op = np.zeros((m, op_h, op_w))
        pad = np.pad(images, ((0, 0), (pad_tb, pad_tb), (pad_lr, pad_lr)),
                                mode='constant')
        for row in range(op_h):
            for col in range(op_w):
                cur = pad[:, row*sh:row*sh+kern1, col*sw:col*sw+kern2]
                op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
        return op
    elif padding == 'valid':
        op_size1 = int(((h - kern1) + 1) / sh)
        op_size2 = int(((w - kern2) + 1) / sw)
        op = np.zeros((images.shape[0], op_size1, op_size2))
        for row in range( op_size1):
            for col in range(op_size2):
                cur = images[:, row*sh:row*sh+kern1, col*sw:col*sw+kern2]
                op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
        return op
    else:
        ph, pw = padding
        new_h = (h - kern1 + (2 * ph) + 1)
        new_w = (w - kern2 + (2 * pw) + 1)
        op = np.zeros((m, new_h, new_w))
        pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')
        for row in range(0, new_h, sh):
            for col in range(0, new_w, sw):
                cur = pad[:, row:row+kern1, col:col+kern2]
                op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
        return op
