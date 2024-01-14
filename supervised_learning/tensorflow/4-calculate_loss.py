#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def calculate_loss(y, y_pred):
    """_summary_
    Args:
        y (_summary_): _summary_
        y_pred (_summary_): _summary_
    Returns:
        _summary_
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
