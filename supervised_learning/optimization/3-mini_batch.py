
#!/usr/bin/env python3
""" module containing function that trains a loaded neural network model
    using mini-batch gradient descent """
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a loaded neural network model using mini-batch gradient descent

        PARAMETERS:
            X_train [np.ndarray]: np array of shape (m, 784) containing
                                    training data
                                    m - number of data points
                                    784 - number of features
            Y_train [np.ndarray]: one-hot encoded np array of shape (m, 10)
                                    containing the training labels
                                    m - same number of data points as in X
                                    ny - number of classes the model should
                                            classify
            X_valid [np.ndarray]: np array of shape (m, 784) containing
                                    the validation data
            Y_valid [np.ndarray]: one-hot encoded np array of shape (m, 10)
                                    containing the validation labels
            batch_size [int]: number of datapoints in a batch
            epochs [int]: number of times the training should pass through
                            the whole dataset
            load_path [str]: path from which to load the model from
            save_path [str]: path to where the model should be saved after training

        RETURNS:
            save_path [str]: path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f'{load_path}.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        t_feed_dict = {x: X_train, y: Y_train}
        v_feed_dict = {x: X_valid, y: Y_valid}
        for i in range(epochs):
            e_t_acc, e_t_loss = sess.run([accuracy, loss], feed_dict=t_feed_dict)
            e_v_acc, e_v_loss = sess.run([accuracy, loss], feed_dict=v_feed_dict)
            print(f'After {i} epochs:')
            print(f'\tTraining Cost: {e_t_loss}')
            print(f'\tTraining Accuracy: {e_t_acc}')
            print(f'\tValidation Cost: {e_v_loss}')
            print(f'\tValidation Accuracy: {e_v_acc}')

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)


            batches = 0
            num_samples = X_train.shape[0]
            last_batch = num_samples % batch_size
            steps =  (num_samples - last_batch) / batch_size

            for j in range(0, int(steps) + 1):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                if j != steps:
                    X_batch = X_shuffled[batches:batches + batch_size]
                    Y_batch = Y_shuffled[batches:batches + batch_size]
                    batches += batch_size
                else:
                    X_batch = X_shuffled[batches:batches + batch_size]
                    Y_batch = Y_shuffled[batches:batches + batch_size]

                # run training operation
                feed_dict = {x: X_batch, y: Y_batch}
                _, acc, los = sess.run([train_op, accuracy, loss], feed_dict=feed_dict)

                if j % 100 == 0:
                    print(f'\tStep {j}:')
                    print(f'\t\tTraining Cost: {los}')
                    print(f'\t\tTraining Accuracy: {acc}')
        e_t_acc, e_t_loss = sess.run([accuracy, loss], feed_dict=t_feed_dict)
        e_v_acc, e_v_loss = sess.run([accuracy, loss], feed_dict=v_feed_dict)
        print(f'After {epochs} epochs:')
        print(f'\tTraining Cost: {e_t_loss}')
        print(f'\tTraining Accuracy: {e_t_acc}')
        print(f'\tValidation Cost: {e_v_loss}')
        print(f'\tValidation Accuracy: {e_v_acc}')
        save_p = saver.save(sess, save_path)
    return save_p 