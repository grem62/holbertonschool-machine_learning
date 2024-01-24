#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def l2_reg_cost(cost):
    """_summary_

    Args:
        cost (_type_): _description_
        lambtha (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_
        m (_type_): _description_
    """
    l2 = tf.losses.get_regularization_losses()
    return cost + l2
