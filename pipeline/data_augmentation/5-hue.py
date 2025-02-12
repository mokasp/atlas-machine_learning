#!/usr/bin/env python3
"""
    Adjusts the hue of an image.

    This function uses TensorFlow's `adjust_hue` method to apply a hue adjustment 
    to the input image. The `delta` value controls the amount of adjustment.

    Args:
        image (Tensor): A 3D tensor representing the image (height, width, channels) with pixel values in the range [0, 1].
        delta (float): The hue adjustment to apply. The value should be in the range of [-1.0, 1.0], where:
            - Positive values will shift the hue towards the "warm" side.
            - Negative values will shift the hue towards the "cool" side.

    Returns:
        Tensor: The image with the adjusted hue.

    Example:
        ```python
        import tensorflow as tf
        image = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=1, dtype=tf.float32)
        adjusted_image = change_hue(image, 0.2)
        ```
"""
import tensorflow as tf


def change_hue(image, delta):
    return tf.image.adjust_hue(image, delta)