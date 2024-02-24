# Keras Example

This repository contains a collection of functions and scripts demonstrating how to build, train, optimize, and evaluate neural network models using the Keras library. The functions provided cover various aspects of deep learning model development, including model architecture design, training, optimization techniques, and saving/loading model configurations and weights.

## Files

### 0-sequential.py
- `build_model(nx, layers, activations, lambtha, keep_prob)`: Builds a neural network with the Keras library.

### 1-input.py
- `build_model(nx, layers, activations, lambtha, keep_prob)`: Builds a neural network with the Keras library without using the `Sequential` class.

### 2-optimize.py
- `optimize_model(network, alpha, beta1, beta2)`: Sets up Adam optimization for a Keras model with categorical crossentropy loss and accuracy metrics.

### 3-one_hot.py
- `one_hot(labels, classes=None)`: Converts a label vector into a one-hot matrix.

### 4-train.py
- `train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False)`: Trains a model using mini-batch gradient descent.

### 5-train.py
- `train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False)`: Trains a model using mini-batch gradient descent and analyzes validation data.

### 6-train.py
- `train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False)`: Trains the model using early stopping based on validation loss.

### 7-train.py
- `train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False)`: Trains the model with learning rate decay.

### 8-train.py
- `train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False)`: Trains the model and saves only the best iteration of the model.

### 9-model.py
- `save_model(network, filename)`: Saves an entire model.
- `load_model(filename)`: Loads an entire model.

### 10-weights.py
- `save_weights(network, filename, save_format='h5')`: Saves a model’s weights.
- `load_weights(network, filename)`: Loads a model’s weights.

### 11-config.py
- `save_config(network, filename)`: Saves a model’s configuration in JSON format.
- `load_config(filename)`: Loads a model with a specific configuration.

### 12-test.py
- `test_model(network, data, labels, verbose=True)`: Tests a neural network.

### 13-predict.py
- `predict(network, data, verbose=False)`: Makes a prediction using a neural network.


## Dependencies
- Python 3.x
- Keras
- NumPy
