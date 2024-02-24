#!/usr/bin/env python3
""" module containing function that converts a label vector into
    a one-hot matrix """


def one_hot(labels, classes=None):
    """ function that converts a label vector into a one-hot matrix

        PARAMETERS
        ==========
        labels [np.ndarray]: label vector to be converted to a one-hot matrix.
        classes [int]: The number of classes.

        RETURNS
        =======
            one-hot matrix of the labels
    """
