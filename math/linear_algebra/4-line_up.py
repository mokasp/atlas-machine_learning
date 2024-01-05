#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    new_arr = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            new_arr.append(arr1[i] + arr2[i])
        return new_arr
