#!/usr/bin/env python3
"""cofacteur"""


def determinant(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose determinant should be calculated

    Returns:
        int: the determinant of matrix
    """
    # Check if matrix is a list of lists and non-empty
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    if len(matrix) != len(matrix[0]) or len(matrix) == 0:
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


def minor(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose minor should be calculated

    Returns:
        int: the minor of matrix
    """
    # Check if matrix is a list of lists and non-empty
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    if len(matrix) != len(matrix[0]) or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]
    minors = []
    for i in range(len(matrix)):
        minors.append([])
        for j in range(len(matrix[0])):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] +
                                                          matrix[i+1:])]
            minors[i].append(determinant(sub_matrix))
    return minors


def cofactor(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose cofactor should be calculated

    Returns:
        int: the cofactor of matrix
    """
    minors = minor(matrix)
    cofactors = []
    for i in range(len(minors)):
        cofactors.append([])
        for j in range(len(minors[0])):
            cofactors[i].append(minors[i][j] * (-1) ** (i + j))
    return cofactors


def adjugate(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose adjugate should be calculated

    Returns:
        int: the adjugate of matrix
    """
    cofactors = cofactor(matrix)
    adjugate = []
    for i in range(len(cofactors)):
        adjugate.append([])
        for j in range(len(cofactors[0])):
            adjugate[i].append(cofactors[j][i])
    return adjugate


def inverse(matrix):
    """_summary_

    Args:
        matrix (list): list of lists whose inverse should be calculated

    Returns:
        int: the inverse of matrix
    """
    adjugate_matrix = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    inverse_matrix = []
    for i in range(len(adjugate_matrix)):
        row = []
        for j in range(len(adjugate_matrix[0])):
            element = adjugate_matrix[i][j] / det
            row.append(element)
        inverse_matrix.append(row)
    return inverse_matrix
