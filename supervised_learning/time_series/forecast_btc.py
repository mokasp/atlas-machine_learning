#!/usr/bin/env python3
""" time series forecasting with an LSTM """
from preprocess_data import *
import tensorflow as tf
import tensorflow.keras as K

class FullDecimal(K.callbacks.Callback):
    """ printing full decimals for losses """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'Epoch {epoch + 1}:', end='')
        for name, value in logs.items():
            print(f'\t{name}={value:.10f}', end='')
        print()

def build_model(seq_len):
    """ construct model """
    LSTM = K.models.Sequential([
        K.layers.LSTM(units=32, input_shape=(seq_len, 1), activation='relu'),
        K.layers.Dense(units=1)])
    return LSTM

def main():
    """ run data processing and training """

    # import the data and clean null values
    df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
    bitcoin_data = clean_data(df)

    # process data and split into train, validation, and test sets
    max_val, train, validate, test = process_data(bitcoin_data)
    train_x, train_y = train
    validate_x, validate_y = validate
    test_x, test_y = test

    # build, compile, and train model
    LSTM = build_model(24)
    LSTM.compile(loss='mean_squared_error', optimizer='adam')
    trained = LSTM.fit(train_x, train_y, epochs=30, batch_size=256, validation_data=(validate_x, validate_y), verbose=False, callbacks=[FullDecimal()])

    print('Test Loss: ', LSTM.evaluate(test_x, test_y))

    prediction = LSTM.predict(test_x)

    visualize_predictions(prediction, test_y, max_val)

if __name__ == '__main__':
    main()