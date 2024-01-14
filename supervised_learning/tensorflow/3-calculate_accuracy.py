#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def calculate_accuracy(y, y_pred):
    """_summary_
    Args:
        y (_summary_): _summary_
        y_pred (_summary_): _summary_
    Returns:
        _summary_
    """
    y_pred = tf.argmax(y_pred, 1)
    y = tf.argmax(y, 1)
    equality = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
