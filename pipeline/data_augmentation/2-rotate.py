#!/usr/bin/env python3
"""
    Rotates an image 90 degrees counterclockwise.

    Args:
        image (Tensor): A 3D tensor representing the image
        (height, width, channels) with pixel values in the range [0, 1].

    Returns:
        Tensor: The image rotated by 90 degrees counterclockwise.

    Example:
        ```
        image = tf.random.uniform(shape=[256, 256, 3], minval=0,
        maxval=1, dtype=tf.float32)
        rotated_image = rotate_image(image)
        ```
"""
import tensorflow as tf


def rotate_image(image):
    """ This function uses TensorFlow's `rot90` method to rotate the input
    image by 90 degrees counterclockwise. The image is rotated in a way
    that its top becomes the left side. """
    return tf.image.rot90(image)
