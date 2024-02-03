#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    Acc = y - y_pred
    return tf.reduce_mean(Acc)