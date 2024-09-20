#!/usr/bin/env python3
""" this script contains a function that performs SARSA(λ) with a gym
    environment

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
            alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
            epsilon_decay=0.05): performs SARSA(λ) on a given environment.
        - ep_greedy_policy(Q, state, epsilon): performs epsilon greedy policy
"""
import gym
import numpy as np


def ep_greedy_policy(Q, state, epsilon):
    """ function that performs the epsilon greedy policy

        Args:
            Q (numpy.ndarray): Q-table where rows represent states and columns
                represent actions.
            state (int): the current state of the agent
            epsilon (float): initial exploration rate for the epsilon-greedy
                policy, default is 1.
    """
    # use epsilon-greedy to decide the next move
    # choose a random value between 0 and 1
    choice = np.random.random()
    # decide to explore or exploit based on the epsilon
    if choice < epsilon:
        # exploration: pick a random action
        action = np.random.randint(len(Q[state]))
    else:
        # exploitation: pick the action with highest q-value
        action = np.argmax(Q[state, :])

    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ function that that performs SARSA(λ)

        Args:
            env (gym.Env): gym environment instance.
            Q (numpy.ndarray): Q-table where rows represent states and columns
                represent actions.
            lambtha (float):  eligibility trace factor
            episodes (int): total number of episodes to train over,
                default is 5000.
            max_steps (int): maximum number of steps per episode,
                default is 100.
            alpha (float): learning rate for Q-learning, default is 0.1.
            gamma (float): discount rate for future rewards, default is 0.99.
            epsilon (float): initial exploration rate for the epsilon-greedy
                policy, default is 1.
            min_epsilon (float): minimum value that epsilon should decay to,
                default is 0.1.
            epsilon_decay (float): rate at which epsilon decays after each
                episode, default is 0.05.

        Returns:
            np.ndarray: Q
                - Q (numpy.ndarray): updated Q-table after sarsa lambda.
    """
    for episode in range(episodes):
        # reset the environment and EoE flag
        state = env.reset()
        end_of_episode = False
        eligibility_trace = np.zeros_like(Q)
        action = ep_greedy_policy(Q, state, epsilon)

        for _ in range(max_steps):

            # use selected action and get new reward
            current_state, reward, end_of_episode, _ = env.step(action)

            # find next action for the td error
            next_action = ep_greedy_policy(Q, current_state, epsilon)

            # calculate the temporal difference error using the immediate
            # reward and the discounted estimate of the future rewards
            td_error = reward + \
                (gamma * Q[current_state, next_action]) - Q[state, action]

            # add one each time a certain state is visited
            eligibility_trace[state, action] += 1

            # update the Q value using the eligibility trace
            Q[state, action] = Q[state, action] + alpha * \
                td_error * eligibility_trace[state, action]

            # decay the eligibility trace
            eligibility_trace *= gamma * lambtha

            # set the new state and action
            state = current_state
            action = next_action

            # check if the episode is complete
            if end_of_episode:
                break

        # epsilon decay - as the agent explores throughout training, we can
        # reduce the epsilon to encourage exploitation
        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
