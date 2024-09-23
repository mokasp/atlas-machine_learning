#!/usr/bin/env python3
""" this script contains functions for implementing a policy gradient agent
    using softmax action selection. The agent learns to select actions based
    on state input and updates its policy weights through reinforcement
    learning.

Dependencies:
    - gym: Standard API for reinforcement learning with a diverse
        collection of reference environments.
    - numpy: A library for numerical operations in Python.

Functions:
    - policy(matrix, weight): Computes action probabilities for a given state.
    - softmax_grad(softmax): Computes the gradient of the softmax function.
    - policy_gradient(state, weight): Computes the action and the gradient
        of the policy for a given state.
"""
import numpy as np
import gym


def policy(matrix, weight):
    """ computes action probabilities for a given state using a linear
        transformation.

    Args:
        matrix (np.ndarray): The input state represented as a matrix.
        weight (np.ndarray): The weight matrix that defines the policy.

    Returns:
        np.ndarray: A probability distribution over actions for the given
            state.
    """
    # apply linear transformation to get raw scores
    raw_scores = np.dot(matrix, weight)
    # apply exponential and shift scores to prevent overflow
    stabilized_scores = np.exp(raw_scores - np.max(raw_scores))
    # convert raw exponentiated scores into probabilities
    action_probs = stabilized_scores / np.sum(stabilized_scores)
    return action_probs


def softmax_grad(softmax):
    """ computes the gradient of the softmax function.

    Args:
        softmax (np.ndarray): The softmax output probabilities.

    Returns:
        np.ndarray: The gradient of the softmax output.
    """

    # reshape softmax probs to prepare for matrix operations
    softmax = softmax.reshape(-1, 1)
    # calculate the softmax gradient by subtracting the outer product from the
    # diagonal matrix
    softmax_grad = np.diagflat(softmax) - np.dot(softmax, softmax.T)
    return softmax_grad


def policy_gradient(state, weight):
    """ computes the action and the gradient of the policy for a given
        state.

    Args:
        state (np.ndarray): current state of the environment.
        weight (np.ndarray): weight matrix for the policy.

    Returns:
        tuple: A tuple containing:
            - action (int): selected action based on the policy.
            - grad (np.ndarray): gradient of the policy for the given state.
    """

    # get action probabilities for the current state using updated policy
    action_probs = policy(state, weight)

    # get next action using action probabilities
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])

    # get gradient of softmax probabilities at the choosen action, which
    # indicates how the action probabilities change
    softmax_g = softmax_grad(action_probs)[action, :]

    # caclulate the log probabilities of the choosen actions based on
    # the softmax gradient,  normalizing the gradient of softmax
    # function relative to the pobability of the choosen action
    log_probs = softmax_g / action_probs[0, action]

    # get the gradient of the policy which will be used to update
    # the policies weights, and add extra dimension to the log probabilities
    grad = state.T.dot(np.expand_dims(log_probs, axis=0))

    return action, grad
