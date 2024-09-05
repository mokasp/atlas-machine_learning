#!/usr/bin/env python3
""" this script contains a function that initializes a Q-table with a
    FrozenLake environment

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - q_init(env): initializes a Q-table with a FrozenLake environment
"""
import numpy as np


def q_init(env):
    """ function that initializes a Q-table with a FrozenLake environment

        Args:
            env (gym environment): FrozenLake instance

        Returns:
            (numpy.ndarray): Empty ndarray of zeros representing the Q-table
    """
    # use the size of the action and space state to  create the empty q-table
    action_space = env.action_space
    state_space = env.observation_space
    return np.zeros((state_space.n, action_space.n))
