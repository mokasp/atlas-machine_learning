#!/usr/bin/env python3
import numpy as np
import gym
import time
from policy_gradient import *


def update_grad(rewards, grads, alpha, gamma, policy_weights):
    for i in range(len(grads)):
        discounted_sum = np.sum(rewards[i:] * (gamma ** rewards[i:]))
        policy_weights += alpha * discounted_sum * grads[i]
    return policy_weights

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]

  policy_weights = np.random.rand(num_states, num_actions)

  scores = []

  for episode in range(nb_episodes + 1):

    state = np.expand_dims(env.reset(), axis=0)

    rewards = []
    grads = []

    score = 0

    for _ in range(500):

      action, grad = policy_gradient(state, policy_weights)

      new_state, reward, done, _ = env.step(action)

      if (episode - 1) % 1000 == 0 and show_result:
        env.render(mode='human')

      state = np.expand_dims(new_state, axis=0)
      score += reward

      rewards.append(reward)
      grads.append(grad)

      if done:
        break
    
    print('Episode: {} Score: {}'.format(episode, score), end="\r", flush=False)
    if episode % 100 == 0:
      time.sleep(.5)

    scores.append(score)
    policy_weights = update_grad(np.array(rewards), grads, alpha, gamma, policy_weights)
  
  return scores