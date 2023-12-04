#!/usr/bin/env python3
"""
    add array
"""


def add_arrays(arr1, arr2):

    if len(arr1) != len(arr2):
        return None
    else:
        new_list = []
        for i in range(len(arr1)):
            new_list.append(int(arr1[i]) + int(arr2[i]))
        return new_list
