#!/usr/bin/env python3
"""
    Randomly adjusts the contrast of an image.

    This function uses TensorFlow's `random_contrast` method to randomly adjust 
    the contrast of the input image within the specified range [lower, upper].

    Args:
        image (Tensor): A 3D tensor representing the image (height, width, channels) with pixel values in the range [0, 1].
        lower (float): The lower bound for the contrast adjustment. The contrast will be randomly adjusted by a factor 
                       greater than or equal to `lower`.
        upper (float): The upper bound for the contrast adjustment. The contrast will be randomly adjusted by a factor 
                       less than or equal to `upper`.

    Returns:
        Tensor: The image with the randomly adjusted contrast.

    Example:
        ```python
        import tensorflow as tf
        image = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=1, dtype=tf.float32)
        adjusted_image = change_contrast(image, 0.5, 1.5)
        ```
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    return tf.image.random_contrast(image, lower, upper)