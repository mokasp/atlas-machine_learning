#!/usr/bin/env python3
import numpy as np
import gym


def policy(matrix, weight):
    # apply linear transformation to get raw scores
    raw_scores = np.dot(matrix, weight)
    # apply exponential and shift scores to prevent overflow
    stabilized_scores = np.exp(raw_scores - np.max(raw_scores))
    # convert raw exponentiated scores into probabilities
    action_probs = stabilized_scores / np.sum(stabilized_scores)
    return action_probs


def softmax_grad(softmax):
    # reshape softmax probs to prepare for matrix operations
    softmax = softmax.reshape(-1, 1)
    # calculate the softmax gradient by subtracting the outer product from the
    # diagonal matrix
    softmax_grad = np.diagflat(softmax) - np.dot(softmax, softmax.T)
    return softmax_grad


def policy_gradient(state, weight):

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
