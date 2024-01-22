#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_:
"""


import numpy as np


def sensitivity(confusion):
    """_summary_

    Args:
        confusion (_type_): 
    """
    sensitiv = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        sensitiv[i] = confusion[i][i] / np.sum(confusion[0])
    return sensitiv
