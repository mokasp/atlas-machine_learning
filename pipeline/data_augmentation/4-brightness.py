#!/usr/bin/env python3
""" This function uses TensorFlow's `random_brightness` method to randomly
    adjust the brightness of the input image by a value within the range
    [-max_delta, max_delta]. """
import tensorflow as tf


def change_brightness(image, max_delta):
    return tf.image.random_brightness(image, max_delta)
