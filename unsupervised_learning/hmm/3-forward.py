#!/usr/bin/env python3
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    
    # reshape initial dist to make compatible with other matrices
    init = Initial.T[0]

    # get shapes
    T = Observation.shape[0]
    N, M = Emission.shape

    # create empty matrix to store forward path probabilities
    F = np.zeros((T, N))

    # get first forward probability with probability of the initial state times the emission probability of first observation
    F[0, :] = init * Emission[:, Observation[0]]

    # for each observation
    for t in range(1, T):

        # iterate through each state
        for j in range(N):
            
            # calculate the forward probaility of that state with that observation
            F[t, j] = F[t - 1].dot(Transition[:, j]) * Emission[j, Observation[t]]
    
    # transform to get correct shape and return
    return F.T