#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""
_summary_
"""


def create_confusion_matrix(labels, logits):
    """_summary_
    Args:
        labels ([type]): [description]
        logits ([type]): [description]
    """
    matrix = np.matmul(labels.T, logits)
    return matrix
