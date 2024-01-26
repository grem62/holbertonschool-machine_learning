#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """_summary_

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        keep_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                 mode="fan_avg")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(rate=keep_prob)
    return dropout(layer(prev))
