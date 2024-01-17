#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """_summary_

    Args:
        loss (_type_): _description_
        alpha (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
        epsilon (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2,
                                  epsilon).minimize(loss)
