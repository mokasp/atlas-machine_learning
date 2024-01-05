#!/usr/bin/env python3
""" module containing function that adds two arrays """


def add_arrays(arr1, arr2):
    """ func that returns the sum of two arrays"""
    new_arr = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            new_arr.append(arr1[i] + arr2[i])
        return new_arr
