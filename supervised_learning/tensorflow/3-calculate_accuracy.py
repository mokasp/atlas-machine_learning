#!/usr/bin/env python3
import tensorflow.compat.v1 as tf

def calculate_accuracy(y, y_pred):
    pred = tf.math.argmax(y_pred, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))
