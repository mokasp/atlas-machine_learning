#!/usr/bin/env python3
""" module that contains function that prepares timeseries data for training """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_data(df):
    """ cleans and refines data """
    # convert unix time to datetime so 60s intervals can be reduce to hour intervals
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df.resample('H').mean()

    # remove 2014 data because it is mostly NaN
    cutoff_date = pd.to_datetime('2015-01-26 07:00:00')
    df = df.loc[cutoff_date:]

    # select only the close column
    bitcoin_data = df.reset_index()
    bitcoin_data = bitcoin_data[['Close']]

    # fill in remaining NaN using linear interpolation
    bitcoin_data = bitcoin_data.interpolate(method='linear')

    return bitcoin_data

def make_sequences(df, seq_len):
    """ creates sequences for time series forcasting """
    x, y = [], []
    for i in range(len(df)- seq_len ):
        # sliding window
        x.append(df.iloc[i:(i+seq_len)].values)
        y.append(df.iloc[i+seq_len])
    return np.array(x), np.array(y)

def process_data(df):
    """ full preprocessing """

    # normalize all values inbetween 0 and 1
    normalized = df.copy()
    max_val = df['Close'].abs().max()
    for column in df.columns:
        normalized[column] = df[column] / df[column].abs().max()
        num_dp = normalized.shape[0]

        # train, validation, and test split
        train = normalized[:int(num_dp * 0.75)]
        validate = normalized[int(num_dp * 0.75):int(num_dp * 0.875)]
        test = normalized[int(num_dp * 0.875):]

        # make sequences
        train_x, train_y = make_sequences(train, 24)
        validate_x, validate_y = make_sequences(validate, 24)
        test_x, test_y = make_sequences(test, 24)


    return max_val, (train_x, train_y), (validate_x, validate_y), (test_x, test_y)

def visualize_predictions(prediction, actual, max_val):
    prediction = prediction.flatten()
    actual = actual.flatten()

    # unnormalize values
    prediction *= max_val
    actual *= max_val

    plt.figure(figsize=(15, 10))
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.title('True vs Predicted Values')
    plt.plot(actual, label='True Closing Price', alpha=0.7)
    plt.plot(prediction, label='Predicted Closing Price', alpha=0.7)
    plt.legend()
    plt.show()