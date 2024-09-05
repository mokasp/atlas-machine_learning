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
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make('BreakoutDeterministic-v4')

memory = SequentialMemory(limit=1000000, window_length=4)

policy = EpsGreedyQPolicy()
env = FrameStack(GrayScaleObservation(ResizeObservation(env, shape=(84, 84))), 4)
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
dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=env.action_space.n, nb_steps_warmup=10000,
                  target_model_update=1e-2, gamma=.99)
dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
dqn.save_weights('policy.h5')
