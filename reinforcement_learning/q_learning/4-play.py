#!/usr/bin/env python3
""" this script contains a function that has the trained Q-learning agent
    play an episode and displays the board states.

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - play(env, Q, max_steps=100): has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """ function that has the trained Q-learning agent play an episode and
        displays the board states.

        Args:
            env (gym.Env): FrozenLake environment instance.
            Q (numpy.ndarray): Q-table where rows represent states and columns
                represent actions.
            max_steps (int): maximum number of steps in the episode, default
                is 100.

        Returns:
            float: total reward received during the episode.
    """
    # reset the environment and start rendering
    state = env.reset()
    env.render()

    for i in range(max_steps):
        # choose action with best Q-Value
        random_action = Q[state].argmax()

        # perform the action and obtain reward
        current_state, reward, end_of_episode, _ = env.step(random_action)

        # set the new state
        state = current_state

        # render for each step and check for end of the episode
        env.render()
        if end_of_episode:
            break
    return reward
