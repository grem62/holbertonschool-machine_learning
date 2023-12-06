#!/usr/bin/env python3

def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.

    Parameters:
    - mat1 (list of lists): The first matrix.
    - mat2 (list of lists): The second matrix.

    Returns:
    - list of lists: The result of adding the matrices, or None if matrices are not the same shape.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    new_matrix = []
    for line in range(len(mat1)):
        line_matrix = []
        for column in range (len(mat1[line])):
            line_matrix.append(mat1[line][column] + mat2[line][column])
        new_matrix.append(line_matrix)
    return new_matrix
