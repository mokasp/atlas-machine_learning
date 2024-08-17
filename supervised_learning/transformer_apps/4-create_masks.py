#!/usr/bin/env python3
""" this script defines functions for creating masks used in Transformer
    models, which are essential for handling padding and sequence masking
    in the model's attention mechanisms.

    Dependencies:
        - tensorflow: A library for machine learning and artificial
              intelligence.

    Functions:
        - padding_masks(input): Creates a padding mask for the given tensor
              where padding positions are marked.
        - create_masks(inputs, targets): Creates masks required for the
              Transformer model, including padding masks and look-ahead masks.
"""
import tensorflow as tf


def padding_masks(input):
    """ creates a padding mask for a given tensor.

        Args:
            input (tf.Tensor): A tensor representing input sequences, where
                padding is assumed to be represented by zeros.

        Returns:
            tf.Tensor: A padding mask tensor with shape
                `[batch_size, 1, 1, seq_len]` where padding positions are
                marked with 1.0.
    """
    padding_mask = tf.cast(tf.math.equal(input, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


def create_masks(inputs, targets):
    """ creates masks for the encoder and decoder in a Transformer model.

        Args:
            inputs (tf.Tensor): The input tensor to the encoder. Padding
                positions in this tensor are masked out.
            targets (tf.Tensor): The target tensor for the decoder. Padding
                and look-ahead masks are applied.

        Returns:
            tuple: A tuple containing three masks:
              - encoder_mask (tf.Tensor): The padding mask for the encoder
                    input, with shape `[batch_size, 1, 1, seq_len]`.
              - combined_mask (tf.Tensor): The combined mask for the decoder
                    targets, incorporating both padding and look-ahead masks,
                    with shape `[batch_size, 1, seq_len, seq_len]`.
              - decoder_mask (tf.Tensor): The padding mask for the decoder
                    input, with shape `[batch_size, 1, 1, seq_len]`.
    """
    encoder_mask = padding_masks(inputs)
    decoder_mask = padding_masks(inputs)

    length = tf.shape(targets)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((length, length)),
                                              -1, 0)
    decoder_target_mask = padding_masks(targets)

    combined_mask = tf.maximum(decoder_target_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
