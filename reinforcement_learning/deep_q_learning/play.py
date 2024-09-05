#!/usr/bin/env python3
import gym
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
import tensorflow.keras as K
from keras import __version__
K.__version__ = __version__
from K.models import Sequential
from K.layers import Dense, Input, Flatten, Conv2D
from K.optimizers import Adam
Adam._name = 'adam'
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# load in breakout environment
env = gym.make('BreakoutDeterministic-v4')

# set up memory for the agent to store its past interactions
memory = SequentialMemory(limit=1000000, window_length=4)

# use greedy policy, always choosing exploitation and choosing the action with the highest q-value
policy = GreedyQPolicy()

# preprocess data by resizing to 84 x 84, grayscaling, and stacking into 4 frames
env = FrameStack(GrayScaleObservation(ResizeObservation(env, shape=(84, 84))), 4)

# build model according to the atari breakout paper
actions = env.action_space.n
model = Sequential([
          Input(shape=(84,84,4,)),
          Conv2D(filters=32, kernel_size=(8,8), activation="relu"),
          Conv2D(filters=64, kernel_size=(4,4), activation="relu"),
          Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
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

# load in the trained weights and test the trained agent
dqn.load_weights('policy.h5')
dqn.test(env, nb_episodes=5, visualize=True)