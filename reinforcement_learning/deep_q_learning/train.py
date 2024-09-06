#!/usr/bin/env python3
""" script that trains a reinforcement learning agent using the DQN algorithm
    and plays an episode of Atari's Breakout using the trained agent.

    Dependencies:
        - gym==0.25.0: Standard API for reinforcement learning with a diverse
            collection of reference environments.
        - h5py==3.10.0: Python interface to the HDF5 binary data format, used
            formanaging large datasets.
        - keras==2.13.1: High-level neural networks API, written in Python
            and capable of running on top of TensorFlow, CNTK, or Theano.
        - keras-rl2==1.0.4: Reinforcement learning library for Keras,
            providing implementations of RL algorithms such as DQN, DDPG, and
            more.
        - Pillow==10.0.1: Python Imaging Library (PIL) fork, which adds
            image processing capabilities to Python.
        - tensorflow==2.13.0: Open-source platform for machine learning with
            a comprehensive ecosystem of tools, libraries, and community
            resources.
        - numpy==1.23.5: Library for numerical operations in Python, providing
            support for large, multi-dimensional arrays and matrices, along
            with mathematical functions to operate on these arrays.
"""
import tensorflow.keras as K
from keras import __version__
K.__version__ = __version__
from K.optimizers import Adam
from K.layers import Dense, Input, Flatten, Conv2D
from K.models import Sequential
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
import gym
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
Adam._name = 'adam'

# load in breakout environment
env = gym.make('BreakoutDeterministic-v4')

# set up memory for the agent to store its past interactions
memory = SequentialMemory(limit=1000000, window_length=4)

# use the epsilon greedy policy
policy = EpsGreedyQPolicy()

# preprocess data by resizing to 84 x 84, grayscaling, and stacking into 4
# frames
env = FrameStack(
    GrayScaleObservation(
        ResizeObservation(
            env, shape=(
                84, 84))), 4)

# build model according to the atari breakout paper
actions = env.action_space.n
model = Sequential([
    Input(shape=(84, 84, 4,)),
    Conv2D(filters=32, kernel_size=(8, 8), activation="relu"),
    Conv2D(filters=64, kernel_size=(4, 4), activation="relu"),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    Flatten(),
    Dense(units=256, activation="relu"),
    Dense(units=actions, activation="linear"),
])

# initialize the agent itself
dqn = DQNAgent(model=model, memory=memory, policy=policy,
               nb_actions=env.action_space.n, nb_steps_warmup=10000,
               target_model_update=1e-2, gamma=.99)

# compile the agent with the adam optimizer
dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

# fit the model and save the weights when training is complete
dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
dqn.save_weights('policy.h5')
