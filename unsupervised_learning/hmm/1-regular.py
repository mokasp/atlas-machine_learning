#!/usr/bin/env python3
import numpy as np


def regular(p):
         
        # check to make sure all values in matrix are positive
        if p.all() > 0:

            # get the eigen values and eigen vectors
            values, vectors = np.linalg.eig(p.T)

            # find the index of the eigenvalue that equals 1
            idx = list(np.round(values, decimals=1)).index(1)

            # get the vector associated with the value of 1
            sstate_prob = vectors[:, idx]

            # normalize so probabilities add to 1
            sstate_prob /= sum(sstate_prob)

            return  np.array([sstate_prob])
        return None