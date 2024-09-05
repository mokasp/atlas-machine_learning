#!/usr/bin/env python3
""""""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
  """"""
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