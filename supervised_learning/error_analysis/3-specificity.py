#!/usr/bin/env python3
"""_summary_
"""


import numpy as np


def specificity(confusion):
    """_summary_

    Args:
        confusion (_type_):
    """
    specif = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        specif[i] = (np.sum(confusion) - np.sum(confusion[i]) -
                     np.sum(confusion[:, i]) + confusion[i][i])
        specif[i] = specif[i] / (np.sum(confusion) - np.sum(confusion[i]))
    return specif
