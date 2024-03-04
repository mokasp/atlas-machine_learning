#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.layers.Conv2D(
        kernel_initializer=init, filters=6, kernel_size=5, padding='same', activation='relu')(x)

    maxpool1 = tf.layers.MaxPooling2D(2, 2)(conv1)

    conv2 = tf.layers.Conv2D(
        kernel_initializer=init, filters=16, kernel_size=5, padding='valid', activation='relu')(maxpool1)

    maxpool2 = tf.layers.MaxPooling2D(2, 2)(conv2)

    maxpool2_flat = tf.layers.Flatten()(maxpool2)

    dense1 =tf.layers.Dense(kernel_initializer=init, units=120, activation='relu')(maxpool2_flat)

    dense2 =tf.layers.Dense(kernel_initializer=init, units=84, activation='relu')(dense1)

    logits = tf.layers.Dense(kernel_initializer=init, units=10)(dense2)

    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=logits
    )

    optim = tf.train.AdamOptimizer()
    train_op = optim.minimize(cost)

    op = tf.nn.softmax(logits)

    pred = tf.math.argmax(logits, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)

    acc = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return op, train_op, cost, acc