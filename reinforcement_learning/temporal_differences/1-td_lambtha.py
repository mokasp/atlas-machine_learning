#!/usr/bin/env python3
""" this script contains a function that performs TD(λ) with a gym
    environment

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
        alpha=0.1, gamma=0.99): performs TD(λ) on a given environment.
"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """ function that performs the TD(λ) algorithm

        Args:
            env (gym.Env): gym environment instance.
            V (numpy.ndarray): value estimate
            policy (function): takes in a state and returns the next action
                to take
            lambtha (float):  eligibility trace factor
            episodes (int): total number of episodes to train over,
                default is 5000.
            max_steps (int): maximum number of steps per episode,
                default is 100.
            alpha (float): learning rate, default is 0.1.
            gamma (float): discount rate for future rewards, default is 0.99.


        Returns:
            np.ndarray: V
                - V (numpy.ndarray): updated value estimate
    """

    for episode in range(episodes):
        # reset the environment and EoE flag
        state = env.reset()
        end_of_episode = False

        # initialize empty array to keep track of which states have been
        # visited and how many times
        eligibilty_trace = np.zeros_like(V)

        for _ in range(max_steps):
            # use policy to decide which action to take
            action = policy(state)

            # use selected action and get new reward
            current_state, reward, end_of_episode, _ = env.step(action)

            # caculate the temporal difference error
            td_error = reward + gamma * V[current_state] - V[state]
            # add one each time a certain state is visited
            eligibilty_trace[state] += 1

            # update the value estimate using the eligibilty trace
            V += alpha * td_error * eligibilty_trace

            # decay the eligibility trace
            eligibilty_trace *= gamma * lambtha

            # go to next state
            state = current_state

            # check if episode is over
            if end_of_episode:
                break

    return V
