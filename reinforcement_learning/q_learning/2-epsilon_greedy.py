#!/usr/bin/env python3
""" this script contains a function that uses the Epsilon-Greedy policy to
    determine which action the agent will take next

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - epsilon_greedy(Q, state, epsilon): uses epsilon-greedy to determine
            the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ function that selects the next action using the epsilon-greedy
        strategy, it decides whether to explore or  exploit based on epsilon

        Args:
            Q (numpy.ndarray): Q-table, where rows represent states and
                columns represent actions.
            state (int): current state of the environment.
            epsilon (float): probability of selecting a random action
                (exploration rate).

        Returns:
            int: the index of the next action the agent will take.
    """
    # use epsilon-greedy to decide the next move
    # choose a random value between 0 and 1
    choice = np.random.random()

    # decide to explore or exploit based on the epsilon
    if choice < epsilon:
        # exploration: pick a random action
        return np.random.randint(len(Q[state]))
    else:
        # exploitation: pick the action with highest q-value
        return np.argmax(Q[state])
