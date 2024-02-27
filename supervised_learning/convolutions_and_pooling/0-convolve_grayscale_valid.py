#!/usr/bin/env python3
""" module containing function that performs a same convolution on grayscale
    images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ function that performs a same convolution on grayscale images

        PARAMETERS
        ==========
            images [np.ndarray]: multiple gray scale images of shape (m, h, w)
                    m - number of images
                    h - height in pixels of the images
                    w - width in pixels of the images
            kernel [np.ndarray]: kernel for the convolution of shape (kh, kw)
                    kh - height of the kernel
                    kw - width of the kernel

        RETURNS
        =======
            numpy.ndarray containing the convolved images
    """
    op_size = (len(images[0]) - kernel.shape[0]) + 1
    op = np.zeros((images.shape[0], op_size, op_size))
    kern = kernel.shape[0]
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    for i in range(m):
        for j in range(h * w):
            row = j // h
            col = j % w
            if row < h - 2 and col < w - 2:
                cur = images[i][row:row+kern, col:col+kern]
                op[i, row, col] = np.sum(np.multiply(cur, kernel))
    return op
