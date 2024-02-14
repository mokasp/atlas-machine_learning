#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
import numpy as np
def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    for lay in range(len(layers)):
        if lay != len(layers) - 1:
            x = create_layer(prev, layers[lay], activations[lay], epsilon)
        else:
            x = create_last_layer(x, layers[lay])
    return x

def create_layer(prev, n, activation, epsilon):
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        name="layer",
        kernel_initializer=init
    )
    x = layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    activated = tf.keras.layers.Activation(activation)
    return activated(norm)

def create_last_layer(prev, n):
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        name="layer",
        kernel_initializer=init
    )
    return layer(prev)


def shuffle_data(X, Y):
    """ shuffles the datapoints in two maticies the same way

         PARAMETERS:
            X [np.ndarray]: first np array of shape (m, nx) to be shuffled
                            m - number of data points
                            nx - number of features in X
            X [np.ndarray]: second np array of shape (m, ny) to be shuffled
                            m - same number of data points as in X
                            ny - number of features in Y

    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1]], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, Y_train.shape[1]], name="y")
    tf.compat.v1.add_to_collection('x', x)
    tf.compat.v1.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.compat.v1.add_to_collection('y_pred', y_pred)


    # intialize loss and add it to collection
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )
    tf.compat.v1.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    pred = tf.math.argmax(y_pred, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    tf.compat.v1.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    steps = round(X_train.shape[0] / batch_size)
    decay_step = steps * epochs


    # create "alpha" the learning rate decay operation in tensorflow
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    optim = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                   beta2=beta2, epsilon=epsilon)
    train_op = optim.minimize(loss, global_step)
    tf.compat.v1.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        t_feed_dict = {x: X_train, y: Y_train}
        v_feed_dict = {x: X_valid, y: Y_valid}
        for i in range(epochs):
            # print training and validation cost and accuracy
            e_t_acc = sess.run(accuracy, feed_dict=t_feed_dict)
            e_t_loss = sess.run(loss, feed_dict=t_feed_dict)
            e_v_acc = sess.run(accuracy, feed_dict=v_feed_dict)
            e_v_loss = sess.run(loss, feed_dict=v_feed_dict)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(e_t_loss))
            print("\tTraining Accuracy: {}".format(e_t_acc))
            print("\tValidation Cost: {}".format(e_v_loss))
            print("\tValidation Accuracy: {}".format(e_v_acc))

            # shuffle data
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