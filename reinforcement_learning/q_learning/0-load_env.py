#!/usr/bin/env python3
""" this script contains a function that initializes a FrozenLake environment
    for a reinforcement learning agent to perfom Q-learning with

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments

    Functions:
        - load_frozen_lake(desc=None, map_name=None, is_slippery=False): initi-
            alizes a Frozen Lake environment using the gym library
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ function that initializes a FrozenLake environment for a reinforcement
        learning agent to perfom Q-learning with

        Args:
            desc (optional, list of lists): custom description of the map to
                load for the environment, or None
            map_name (optional, string): premade map to load, or None
            is_slippery (optional, bool): determines if the ice is slippery,
                default is False

        Returns:
            The FrozenLake environment
    """
    # use gym to initialize the frozenlake environ
    return gym.make('FrozenLake-v0',
                    desc=desc,
                    map_name=map_name,
                    is_slippery=is_slippery)
