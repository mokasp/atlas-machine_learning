#!/usr/bin/env python3
""" module containing function that trains a model using mini-batch gradient
    descent, learning rate decay, analyzes validation data, performs early
    stopping and also saves the best iteration of the model """


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """ function that trains a model using mini-batch gradient descent,
        learning rate decay, analyzes validation data, performs earlystopping
        and also saves the best iteration of the model

    PARAMETERS
    ==========
        network [keras model]: The model to train.
        data [np.ndarray]: The input data (m, nx).
        labels [np.ndarray]: The labels of data (m, classes).
        batch_size [int]: The size of the batch used for mini-batch gradient
                            descent.
        epochs [int]: The number of passes through data for mini-batch
                        gradient descent.
        verbose [boolean]: Determines if output should be printed during
                            training.
        shuffle [boolean]: Determines whether to shuffle the batches every
                            epoch.
        validation_data [np.ndarray]: Data to validate the model with.
        early_stopping [boolean]: Indicates whether early stopping should
                                    be used.
        patience [int]: The patience used for early stopping.
        learning_rate_decay [boolean]: Indicates whether learning rate decay
                                        should be used.
        alpha [float]: The initial learning rate.
        decay_rate [float]: The decay rate.
        save_best [boolean]: Indicates whether to save the model after each
                                epoch if it is the best.
        filepath [str]: The file path where the model should be saved.

    RETURNS
    =======
        History object
    """
