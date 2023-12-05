#!/usr/bin/env python3
"""
7. Gettin’ Cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    new_matrix = []
    if len(mat1[0]) == len(mat2[0]) and axis == 0:
        new_matrix = mat1 + mat2
        return new_matrix
    if len(mat1) == len(mat2) and axis == 1:
        for i in range(len(mat1)):
            new_matrix.append(mat1[i] + mat2[i])
        return new_matrix
    else:
        return None
