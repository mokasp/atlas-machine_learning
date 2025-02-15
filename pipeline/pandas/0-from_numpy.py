#!/usr/bin/env python3
""""""
import pandas as pd
import string


def from_numpy(array):
    """"""
    num_columns = len(array[0])
    columns = list(string.ascii_uppercase)[:num_columns]
    dataframe = pd.DataFrame(data=array, columns=columns)
    return dataframe