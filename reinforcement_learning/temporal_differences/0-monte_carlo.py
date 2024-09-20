#!/usr/bin/env python3
""" this script contains a function that performs monte carlo algorithm
    with a gym environment

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - monte_carlo(env, V, policy, episodes=5000, max_steps=100,
            alpha=0.1, gamma=0.99): performs monte carlo algorithm on a given
            environment.
"""
import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """ function that performs the monte carlo algorithm

        Args:
            env (gym.Env): gym environment instance.
            V (numpy.ndarray): value estimate
            policy (function): takes in a state and returns the next action
                to take
            episodes (int): total number of episodes to train over,
                default is 5000.
            max_steps (int): maximum number of steps per episode,
                default is 100.
            alpha (float): learning rate for value estimate update,
                default is 0.1.
            gamma (float): discount rate for future rewards, default is 0.99.


        Returns:
            np.ndarray: V
                - V (numpy.ndarray): updated value estimate
    """

    for i in range(episodes):
        # reset environment
        state = env.reset()

        # keep track of each state and its reward
        states = []
        rewards = []
        done = False

        for _ in range(max_steps):
            # get new action
            action = policy(state)

            # take a step and document the states and rewards
            next_state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)

            # go to next state
            state = next_state

            # check if episode is finished
            if done:
                break

        # keep track of the accumulated reward
        returns = 0
        for step in range(len(states) - 1, -1, -1):

            # get the state and reward from this timestep for this episode
            state = states[step]
            reward = rewards[step]

            # calculate accumulated reward
            returns = (gamma * returns) + reward

            # if the state hasnt been visited yet in this episode
            if state not in states[:i]:

                # update the value estimate
                V[state] += alpha * (returns - V[state])
    return V
