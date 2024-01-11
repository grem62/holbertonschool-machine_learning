#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def create_layer(prev, n, activation):
    """_summary_

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_

    Returns:
        _type_: _description_
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init, name='layer')

    return layer(prev)
