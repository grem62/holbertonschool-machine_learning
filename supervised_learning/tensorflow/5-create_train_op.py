#!/usr/bin/env python3
""" _summary_"""


import tensorflow.compat.v1 as tf
"""_summary_
"""


def create_train_op(loss, alpha):
    """_summary_
    Args:
        loss (_summary_): _summary_
        alpha (_summary_): _summary_
    Returns:
        _summary_
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
