#!/usr/bin/env python3
""""""
import numpy as np


def q_init(env):
  """"""
  action_space = env.action_space
  state_space = env.observation_space
  return np.zeros((state_space.n, action_space.n))