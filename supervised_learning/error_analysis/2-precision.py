#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def precision(confusion):
    """_summary_

    Args:
        confusion (_type_):
    """
    precisi0n = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        precisi0n[i] = confusion[i][i] / np.sum(confusion[:, i] + 1e-5 - 1e-5)
    return precisi0n
