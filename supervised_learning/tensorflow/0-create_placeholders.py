#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")
    return x, y