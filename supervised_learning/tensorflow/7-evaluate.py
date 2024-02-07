#!/usr/bin/env python3
""" module cantaining function that evaluates the output of a neural network"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ function that evaluates the output of a neural network"""
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    y_pred = tf.get_collection("y_pred")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model = saver.restore(sess, save_path)
    y_p, acc, los = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})
    sess.close()
    return y_p, acc, los
