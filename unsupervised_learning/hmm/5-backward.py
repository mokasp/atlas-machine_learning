#!/usr/bin/env python3
import numpy as np


def backward(Observation, Emission, Transition, Initial):

    # reshape initial dist to make compatible with other matrices
    init = Initial.T[0]

    # get shapes
    T = Observation.shape[0]
    N, M = Emission.shape

    # create empty matrix to store backward path probabilities
    B = np.zeros((T, N))

    # set last backward probability with ones
    B[T - 1] = np.ones((N))

    # for each observation, iterating backwards
    for t in range(T - 2, -1, -1):

        # iterate through each state
        for j in range(N):

            # calculate the backward probaility of that state with that observation
            B[t, j] = (B[t + 1] * Emission[:, Observation[t + 1]]).dot(Transition[j, :])

    # get likelihood of the observations by summing the initial state times the first observation times the backward probabilities from the first timestep
    likelihood = np.sum(init * Emission[:, Observation[0]] * B[0, :])

    # transform to get correct shape and return
    return likelihood, B.T