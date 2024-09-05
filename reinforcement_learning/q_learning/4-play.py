#!/usr/bin/env python3
""""""
import numpy as np


def play(env, Q, max_steps=100):
  """"""
  # reset the environment and start rendering
  state = env.reset()
  env.render()

  for _ in range(max_steps):
    # choose action with best Q-Value
    random_action = Q[state].argmax()

    # perform the action and obtain reward
    current_state, reward, end_of_episode, _ = env.step(random_action)

    # set the new state
    state = current_state

    # render for each step and check for end of the episode
    env.render()
    if end_of_episode:
      break
  return reward