#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """_summary_

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        lambtha (_type_): _description_
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    reg = tf.keras.regularizers.l2(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, kernel_regularizer=reg)
    return layer(prev)
