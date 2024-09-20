#!/usr/bin/env python3
import gym
import numpy as np


def ep_greedy_policy(Q, state, epsilon):
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

    return action

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):

    for episode in range(episodes):
        # reset the environment and EoE flag
        state = env.reset()
        end_of_episode = False
        eligibility_trace = np.zeros_like(Q)
        action = ep_greedy_policy(Q, state, epsilon)

        for _ in range(max_steps):

            # use selected action and get new reward
            current_state, reward, end_of_episode, _ = env.step(action)

            # find next action for the td error
            next_action = ep_greedy_policy(Q, current_state, epsilon)

            # calculate the temporal difference error using the immediate
            # reward and the discounted estimate of the future rewards
            td_error = reward + (gamma * Q[current_state, next_action]) - Q[state, action]

            # add one each time a certain state is visited
            eligibility_trace[state, action] += 1

            # update the Q value using the eligibility trace
            Q[state, action] = Q[state, action] + alpha * td_error * eligibility_trace[state, action]

            # decay the eligibility trace
            eligibility_trace *= gamma * lambtha

            # set the new state and action
            state = current_state
            action = next_action

            # check if the episode is complete
            if end_of_episode:
                break

        # epsilon decay - as the agent explores throughout training, we can
        # reduce the epsilon to encourage exploitation
        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)


    return Q