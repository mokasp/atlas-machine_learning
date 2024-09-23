#!/usr/bin/env python3
import numpy as np
import gym


def policy(matrix, weight):
    raw_scores = np.dot(matrix, weight)
    stabilized_scores = np.exp(raw_scores - np.max(raw_scores))
    action_probs = stabilized_scores / np.sum(stabilized_scores)
    return action_probs

def softmax_grad(softmax):
    softmax = softmax.reshape(-1,1)
    softmax_grad = np.diagflat(softmax) - np.dot(softmax, softmax.T)
    return softmax_grad

def policy_gradient(state, weight):
    action_probs = policy(state, weight)
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])
    softmax_g = softmax_grad(action_probs)[action, :]
    log_probs = softmax_g / action_probs[0, action]
    grad = state.T.dot(np.expand_dims(log_probs, axis=0))
    return action, grad
