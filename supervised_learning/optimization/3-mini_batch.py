#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        t_feed_dict = {x: X_train, y: Y_train}
        v_feed_dict = {x: X_valid, y: Y_valid}
        for i in range(epochs):
            e_t_acc = sess.run(accuracy, feed_dict=t_feed_dict)
            e_t_loss = sess.run(loss, feed_dict=t_feed_dict)
            e_v_acc = sess.run(accuracy, feed_dict=v_feed_dict)
            e_v_loss = sess.run(loss, feed_dict=v_feed_dict)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(e_t_loss))
            print("\tTraining Accuracy: {}".format(e_t_acc))
            print("\tValidation Cost: {}".format(e_v_loss))
            print("\tValidation Accuracy: {}".format(e_v_acc))

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            batches = 0
            num_samples = X_train.shape[0]
            last_batch = num_samples % batch_size
            if last_batch == 0:
                last_batch = batch_size
            steps = (num_samples - last_batch) / batch_size

            for j in range(0, int(steps) + 1):
                if j != steps:
                    X_batch = X_shuffled[batches:batches + batch_size]
                    Y_batch = Y_shuffled[batches:batches + batch_size]
                    batches += batch_size
                else:
                    X_batch = X_shuffled[batches:batches + last_batch]
                    Y_batch = Y_shuffled[batches:batches + last_batch]

                feed_dict = {x: X_batch, y: Y_batch}
                sess.run(train_op, feed_dict=feed_dict)
                acc = sess.run(accuracy, feed_dict=feed_dict)
                los = sess.run(loss, feed_dict=feed_dict)

                if j % 100 == 0:
                    print("\tStep {}:".format(j))
                    print("\t\tTraining Cost: {}".format(los))
                    print("\t\tTraining Accuracy: {}".format(acc))
        e_t_acc = sess.run(accuracy, feed_dict=t_feed_dict)
        e_t_loss = sess.run(loss, feed_dict=t_feed_dict)
        e_v_acc = sess.run(accuracy, feed_dict=v_feed_dict)
        e_v_loss = sess.run(loss, feed_dict=v_feed_dict)
        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(e_t_loss))
        print("\tTraining Accuracy: {}".format(e_t_acc))
        print("\tValidation Cost: {}".format(e_v_loss))
        print("\tValidation Accuracy: {}".format(e_v_acc))
        save_p = saver.save(sess, save_path)
    return save_p
