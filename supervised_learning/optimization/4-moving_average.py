#!/usr/bin/env python3
""" module containing function that calculates the weighted moving
    average of a data set """
import numpy as np


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set

        PARAMETERS:
            data [list]: data to calculate the moving average
            beta [float]: weight used for the moving average

        RETURNS:
            averages [list]: the moving averages of data

    """
    m_a = []
    for dp in range(len(data) + 1):
        weights = []
        m_a.append(0)
        for idx in reversed(range(dp)):
            weights.append(beta ** idx)
        s_o_w = sum(weights)
        for idx in range(len(weights)):
            weights[idx] = weights[idx] / s_o_w
        for idx in range(dp):
            m_a[dp] += weights[idx] * data[idx]
    return m_a[1:]
