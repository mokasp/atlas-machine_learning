# Deep Q-Learning

this project focuses on training a Deep Q-Network (DQN) agent to play Atari's Breakout using keras-rl2, a reinforcement learning library for Keras, and gym, an API that provides a diverse collection of environments for RL research. it demonstrates how to integrate these powerful tools to train an agent that learns to play the game autonomously through trial and error.

## Key Concepts
- Environment: atari’s Breakout is used as the environment, where the agent learns to control the paddle and break the bricks by interacting with the game state and receiving rewards based on performance
- Agent: utilizing a DQN agent from keras-rl2. Deep Q-network (DQN) is a reinforcement learning algorithm that  uses a neural network to approximate the Q-function, which predicts the expected future reward of taking a certain action in a given state.
- Memory: the agent uses SequentialMemory to store experiences
- Epsilon-Greedy Policy: The agent randomly explores different actions with a probability ε to discover better strategies and exploits the best-known actions with a probability 1-ε.


## Dependencies
- gym==0.25.0: Standard API for reinforcement learning with a diverse collection of reference environments.
- h5py==3.10.0: Python interface to the HDF5 binary data format, used formanaging large datasets.
- keras==2.13.1: High-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
- keras-rl2==1.0.4: Reinforcement learning library for Keras, providing implementations of RL algorithms such as DQN, DDPG, and more.
- Pillow==10.0.1: Python Imaging Library (PIL) fork, which adds image processing capabilities to Python.
- tensorflow==2.13.0: Open-source platform for machine learning with a comprehensive ecosystem of tools, libraries, and community resources.
- numpy==1.23.5: Library for numerical operations in Python, providing support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
  

## Scripts
- train.py - trains a reinforcement learning agent using the DQN algorithm and plays an episode of Atari's Breakout using the trained agent
- play.py - allows trained reinforcement learning agent to play an episode of Atari's Breakout and displays the game play.
