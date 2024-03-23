#!/usr/bin/env python3

"""_summary_"""


def determinant(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose determinant should be calculated

    Returns:
        int: the determinant of matrix
    """
    # Check if matrix is a list of lists and non-empty
    if not isinstance(
        matrix, list) or not matrix or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    # Check if matrix is square and non-empty
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, num in enumerate(matrix[0]):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += num * (-1) ** i * determinant(sub_matrix)
    return det
