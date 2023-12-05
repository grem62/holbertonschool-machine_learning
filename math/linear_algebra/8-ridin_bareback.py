#!/usr/bin/env python3
"""
matrix multiplication
"""


def mat_mul(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
    """
    if len(mat1[0]) != len(mat2):
        return None
    new_matrice = []
    for row in range(len(mat1)):
        line_row = []
        for column2 in range(len(mat2[0])):
            sum = 0
            for column1 in range(len(mat1[0])):
                sum += mat1[row][column1] * mat2[column1][column2]
            line_row.append(sum)
        new_matrice.append(line_row)
    return new_matrice
