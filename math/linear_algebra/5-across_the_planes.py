#!/usr/bin/env python3
"""
additional matrix
"""


def add_matrices2D(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_matrix = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        new_matrix.append(row)

    return new_matrix
