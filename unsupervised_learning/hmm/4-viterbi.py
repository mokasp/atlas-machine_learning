#!/usr/bin/env python3
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    
    # reshape initial dist to make compatible with other matrices
    init = Initial.T[0]

    # get shapes
    T = Observation.shape[0]
    N, M = Emission.shape

    # create empty arrays
    V = np.zeros((N, T))
    M = np.zeros((N, T))
    best_path = np.zeros(T, dtype=int)

    # initialize first prob
    V[:, 0] = init * Emission[:, Observation[0]]

    # for each observation
    for t in range(1, T):

        # iterate through each state
        for j in range(N):
            
            # get probability to use to find max's
            prob = V[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]]

            # find max among the state probs
            V[j, t] = np.max(prob)

            # get indexs of the max to store the state
            M[j, t] = np.argmax(prob)

    # get last state
    final = np.argmax(V[:, -1])
    
    # get last prob in last determined state
    P = V[final, -1]

    # add last determined state to best_path
    best_path[-1] = final

    # back tracking
    for t in range(T  - 2, -1, -1):

        # add appropriate state at each time t to best_path
        best_path[t] = M[best_path[t + 1], t + 1]
    
    return list(best_path), P