#!/usr/bin/env python3
"""_summary_
"""

import tensorflow as tf
"""_summary_
"""


def create_placeholders(nx, classes):
    """_summary_

    Args:
        nx (_type_): _description_
        classes (_type_): _description_
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
