#!/usr/bin/env python3
""" module containing function that trains a model using mini-batch
    gradient descent """


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """ function that trains a model using mini-batch gradient descent

        PARAMETERS
        ==========
            network [keras model]: the model to train
            data [np.ndarray]: the input data (m, nx)
            labels [np.ndarray]: the labels of data (m, classes)
            batch_size [int]: the size of the batch used for mini-batch
                                gradient descent
            epochs [int]: the number of passes through data for mini-batch
                            gradient descent
            verbose [boolean]: determines if output should be printed during
                                training
            shuffle [boolean]: determines whether to shuffle the batches
                                every epoch.

        RETURNS
        =======
            History object
    """
