#!/usr/bin/env python3
""" this script implements a policy gradient reinforcement learning agent
    for the CartPole environment using the OpenAI Gym. The agent learns
    to select actions based on state input and updates its policy weights
    through the calculation of gradients from the rewards received.

Dependencies:
    - gym: Standard API for reinforcement learning with a diverse
        collection of reference environments.
    - numpy: A library for numerical operations in Python.
    - time: A library for time-related functions.

Functions:
    - update_grad(rewards, grads, alpha, gamma, policy_weights):
        updates the policy weights based on the rewards and gradients
        collected during an episode.

    - train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
        trains the policy gradient agent by running multiple episodes
        in the given environment. Optionally renders the environment
        to visualize the agent's performance.
"""
import numpy as np
import gym
import time
from policy_gradient import *


def update_grad(rewards, grads, alpha, gamma, policy_weights):
    """ updates the policy weights based on the rewards and gradients
        collected during an episode.

    Args:
        rewards (np.ndarray): rewards received for each action taken.
        grads (list): list of gradients corresponding to each action taken.
        alpha (float): learning rate for updating the weights.
        gamma (float): discount factor for future rewards.
        policy_weights (np.ndarray): current weights of the policy.

    Returns:
        np.ndarray: updated policy weights.
    """
    # over each timesteps gradient
    for i in range(len(grads)):
        # find the discounted sum of rewards
        discounted_sum = np.sum(rewards[i:] * (gamma ** rewards[i:]))
        # update the policy weights proportioanl to the rewards and learning
        # rate
        policy_weights += alpha * discounted_sum * grads[i]
    return policy_weights


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ trains the policy gradient agent by running multiple episodes
        in the given environment and optionally renders the environment
        to visualize the agent's performance.

    Args:
        env (gym.Env): OpenAI Gym environment to train the agent in.
        nb_episodes (int): number of episodes to run for training.
        alpha (float): learning rate for updating the policy weights.
        gamma (float): discount factor for future rewards.
        show_result (bool): if True, renders the environment every 1000
            episodes.

    Returns:
        list: list of scores for each episode.
    """
    # initialize policy weights using the action and observation space
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    policy_weights = np.random.rand(num_states, num_actions)

    # keep track of scores for each episode
    scores = []

    # for each episode
    for episode in range(nb_episodes + 1):

        # reset environment to get initial state
        # add extra dimention to the state
        state = np.expand_dims(env.reset(), axis=0)

        # keep track of rewards and gradients for the episode
        rewards = []
        grads = []

        # keep track of score during this episode
        score = 0

        # max score for cartpole v1 is 500, so each episode can run for 500
        # steps maximum
        for _ in range(500):

            # get new action and the policy gradient for the update
            action, grad = policy_gradient(state, policy_weights)

            # get new state and reward
            new_state, reward, done, _ = env.step(action)

            # render cartpole every 1000 episodes
            if (episode - 1) % 1000 == 0 and show_result:
                env.render(mode='human')

            # add new dimension to new state and add reward to the score
            state = np.expand_dims(new_state, axis=0)
            score += reward

            # keep track of the rewards and gradients for each episode
            rewards.append(reward)
            grads.append(grad)

            # if game is terminated, break
            if done:
                break

        # print the episode and score for every episode, occasionally pausing
        # to better view score
        print(
            'Episode: {} Score: {}'.format(
                episode,
                score),
            end="\r",
            flush=False)
        if episode % 100 == 0:
            time.sleep(.5)

        # keep track of each total score for an episode
        scores.append(score)
        # update the policy weights
        policy_weights = update_grad(
            np.array(rewards), grads, alpha, gamma, policy_weights)

    return scores
