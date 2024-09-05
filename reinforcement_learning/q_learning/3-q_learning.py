#!/usr/bin/env python3
""" this script contains a function that performs Q-learning with a FrozenLake
    environment

    Dependencies:
        - gym: standard API for reinforcement learning with a diverse
            collection of reference environments
        - numpy: A library for numerical operations in Python.

    Functions:
        - epsilon_greedy(Q, state, epsilon): trains a Q-learning agent on a
            given environment
"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ function that trains a Q-learning agent on a given environment.
        
        Args:
            env (gym.Env): FrozenLake environment instance.
            Q (numpy.ndarray): Q-table where rows represent states and columns represent actions.
            episodes (int): total number of episodes to train over, default is 5000.
            max_steps (int): maximum number of steps per episode, default is 100.
            alpha (float): learning rate for Q-learning, default is 0.1.
            gamma (float): discount rate for future rewards, default is 0.99.
            epsilon (float): initial exploration rate for the epsilon-greedy policy, default is 1.
            min_epsilon (float): minimum value that epsilon should decay to, default is 0.1.
            epsilon_decay (float): rate at which epsilon decays after each episode, default is 0.05.
        
        Returns:
            tuple: (Q, total_rewards)
                - Q (numpy.ndarray): updated Q-table after training.
                - total_rewards (list): list containing the total reward per episode.
    """
    # keep track of reward at end of each episode
    all_rewards = []

    for episode in range(episodes):
        # reset the environment and EoE flag
        state = env.reset()
        end_of_episode = False

        # combined reward for each episode
        rewards = 0
        for _ in range(max_steps):
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

        # use selected action and get new reward
        current_state, reward, end_of_episode, _ = env.step(action)
        # if the agent fell in a hole, subtract from the total reqard
        if end_of_episode and reward == 0:
            reward = -1
        
        # calculate the temporal difference error using the immediate reward
        # and the discounted estimate of the future rewards
        td_error = (reward + gamma * np.max(Q[current_state, :]) - Q[state, action])
        
        # update the Q value, set new state, and add new reward to the total
        Q[state, action] = Q[state, action] + alpha * td_error
        state = current_state
        rewards += reward

        # check if the episode is complete
        if end_of_episode:
            break
        
        # epsilon decay - as the agent explores throughout training, we can reduce
        # the epsilon to encourage exploitation
        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)

        # keep track of each reward total
        all_rewards.append(rewards)

    return Q, all_rewards