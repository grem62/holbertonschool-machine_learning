#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""
import
"""


def sensitivity(confusion):
    """_summary_

    Args:
        confusion (_type_): _description_
    """
    sensibilité = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        sensibilité[i] = confusion[i][i] / np.sum(confusion[0])
    return sensibilité
