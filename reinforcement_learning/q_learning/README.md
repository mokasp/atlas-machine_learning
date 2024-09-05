# Q-Learning

this project explores Reinforcement Learning (RL) techniques using OpenAI's Gym environment, specifically the FrozenLake environment. FrozenLake is a classic RL environment that simulates an agent navigating a grid world with the goal of reaching a designated goal state while avoiding holes. It serves as an excellent introduction to RL algorithms and their applications.

## Objectives
- implement and train a Q-learning agent to solve the FrozenLake environment.
- evaluate the performance of the agent in terms of its ability to successfully navigate to the goal state.
- visualize and analyze the agent's learning process and performance metrics.


## Dependencies
- gym==0.13.0: For creating and interacting with the FrozenLake environment.
- numpy: For numerical operations and managing the Q-table.
  

## Functions
- load_frozen_lake(desc=None, map_name=None, is_slippery=False): initializes a Frozen Lake environment using the gym library
- q_init(env): initializes a Q-table with a FrozenLake environment
- epsilon_greedy(Q, state, epsilon): Chooses the next action based on the epsilon-greedy strategy.
- train(env, Q, episodes, max_steps, alpha, gamma, epsilon, min_epsilon, epsilon_decay): Trains the Q-learning agent on the FrozenLake environment.
- play(env, Q, max_steps): Runs a test episode with the trained agent and displays the environment state.