#!/usr/bin/env python3
"""_summary_
"""


import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


def normalization_constants(X):
    """_summary_

    Args:
        X (_type_): _description_
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
