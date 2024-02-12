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
    for dp in range(len(data)):
        if dp == 0:
            cur_value = 0
        cur_value = (beta * cur_value) + (1 - beta) * data[dp]
        bias = (1 - (beta ** (dp + 1)))
        m_a.append(cur_value / bias)
    return m_a

