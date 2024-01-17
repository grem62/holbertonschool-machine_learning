#!/usr/bin/env python3
"""_summary_"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """_summary_

    Args:
        loss (_type_): _description_
        alpha (_type_): _description_
        beta2 (_type_): _description_
        epsilon (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)
