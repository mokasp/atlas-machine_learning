#!/usr/bin/env python3
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):

    for episode in range(episodes):
        # reset the environment and EoE flag
        state = env.reset()
        end_of_episode = False

        # initialize empty array to keep track of which states have been visited and how many times
        eligibilty_trace = np.zeros_like(V)

        for _ in range(max_steps):
            # use policy to decide which action to take 
            action = policy(state)

            # use selected action and get new reward
            current_state, reward, end_of_episode, _ = env.step(action)

            # caculate the temporal difference error
            td_error = reward + gamma * V[current_state] - V[state]
            # add one each time a certain state is visited
            eligibilty_trace[state] += 1

            # update the value estimate using the eligibilty trace
            V += alpha * td_error * eligibilty_trace

            # decay the eligibility trace
            eligibilty_trace *= gamma * lambtha

            # go to next state
            state = current_state

            # check if episode is over
            if end_of_episode:
                break

    return V