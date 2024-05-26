#!/usr/bin/env python3
import numpy as np


def markov_chain(P, s, t=1):
    n = len(s[0])
    states = np.arange(n)
    state = np.random.choice(states, p=s[0])

    dist = []

    for i in range(t):
        current = state

        p_dist = P[list(states).index(current)]
        
        dist.append(p_dist)
        state = np.random.choice(states, p=p_dist)
    return np.array([np.mean(dist, axis=0)])