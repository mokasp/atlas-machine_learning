#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

def create_train_op(loss, alpha):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train = optimizer.minimize(loss)
    return train