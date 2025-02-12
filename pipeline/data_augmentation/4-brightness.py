#!/usr/bin/env python3

""" this function uses TensorFlow's random_brightness function to randomly
adjust the brightness of the input image by a value within the range
[-max_delta, max_delta] """

import tensorflow as tf


def change_brightness(image, max_delta):
    return tf.image.random_brightness(image, max_delta)
