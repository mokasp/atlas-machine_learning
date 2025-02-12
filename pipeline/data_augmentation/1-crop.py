#!/usr/bin/env python3
"""
    Randomly crops an image to a specified size.

    This function uses TensorFlow's `random_crop` method to randomly crop the input image to the given size. 
    The size should be specified as a list or tuple of the form [height, width, channels].

    Args:
        image (Tensor): A 3D tensor representing the image (height, width, channels) with pixel values in the range [0, 1].
        size (list or tuple): A list or tuple specifying the target size of the cropped image, in the format [height, width, channels].

    Returns:
        Tensor: The image cropped to the specified size.

    Example:
        ```python
        import tensorflow as tf
        image = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=1, dtype=tf.float32)
        cropped_image = crop_image(image, [128, 128, 3])
        ```
"""
import tensorflow as tf


def crop_image(image, size):
    return tf.image.random_crop(image, size)