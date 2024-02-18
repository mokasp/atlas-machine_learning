#!/usr/bin/env python3
import numpy as np
""" module containing function that creates a confusion matrix and a function
    that decodes a one-hot array """


def create_confusion_matrix(labels, logits):
    """ function that creates a confusion matrix


        PARAMETERS
        ==========
        labels [np.ndarray]: one-hot matrix of shape (m, class) containing
                                the correct labels for each data point
                                    > m [int]:  number of data points
                                    > classes [int]: number of classes

        logits [np.ndarray]: one-hot matrix of shape (m, class) containing
                                the logitsicted labels
                                    > m [int]:  number of data points
                                    > classes [int]: number of classes


        RETURNS
        =======
        [np.ndarray]: confusion array of shape (classes, classes) with row
                        indices representing the correct labels and column
                        indices representing the logitsicted labels
    """
    classes = np.arange(0, labels.shape[1], 1)
    labels = decode(labels)
    logits = decode(logits)
    confusion = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            confusion[i, j] = np.sum((labels == classes[i]) &
                                     (logits == classes[j]))

    return confusion


def decode(one_hot):
    """ function that decodes a one-hot array """
    decoded = []
    for i in range(one_hot.shape[0]):
        for j in range(one_hot.shape[1]):
            if one_hot[i][j] == 1:
                decoded.append(j)

    return decoded
