#!/usr/bin/env python3
""" module cantaining function that evaluates the output of a neural network"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ function that evaluates the output of a neural network"""
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    graph = tf.get_default_graph()
    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = saver.restore(sess, save_path)
        return sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})
