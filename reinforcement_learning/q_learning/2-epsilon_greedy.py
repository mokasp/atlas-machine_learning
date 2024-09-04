#!/usr/bin/env python3
""""""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
  choice = np.random.random()
  if choice < epsilon:
    return np.random.randint(len(Q[state]))
  else:
    return np.argmax(Q[state])