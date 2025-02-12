#!/usr/bin/env python3
"""
    Flips an image horizontally (left to right).

    This function uses TensorFlow's `flip_left_right` method to flip the input image along the vertical axis, 
    creating a mirror image of the original.

    Args:
        image (Tensor): A 3D tensor representing the image (height, width, channels) with pixel values in the range [0, 1].

    Returns:
        Tensor: The image flipped horizontally.

    Example:
        ```python
        import tensorflow as tf
        image = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=1, dtype=tf.float32)
        flipped_image = flip_image(image)
        ```
"""
import tensorflow as tf


def flip_image(image):
    return tf.image.flip_left_right(image)
