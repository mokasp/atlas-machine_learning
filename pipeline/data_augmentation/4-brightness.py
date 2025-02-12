#!/usr/bin/env python3
"""
    Randomly adjusts the brightness of an image.

    Args:
        image (Tensor): A 3D tensor representing the image (height, width, channels) with pixel values in the range [0, 1].
        max_delta (float): The maximum brightness adjustment. The brightness will be randomly adjusted by a factor 
                           between [-max_delta, max_delta].

    Returns:
        Tensor: The image with the randomly adjusted brightness.

    Example:
        ```
        image = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=1, dtype=tf.float32)
        adjusted_image = change_brightness(image, 0.2)
        ```
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """ this function uses TensorFlow's random_brightness function to randomly
    adjust the brightness of the input image by a value within the range
    [-max_delta, max_delta] """
    return tf.image.random_brightness(image, max_delta)
