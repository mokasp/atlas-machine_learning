#!/usr/bin/env python3
import tensorflow.compat.v1 as tf

def calculate_accuracy(y, y_pred):
    return tf.reduce_mean(tf.argmax(y, axis=1) - tf.argmax(y_pred, axis=1))