#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """_summary_

    Args:
        A_prev (_type_): _description_
        filters (_type_): _description_

    Returns:
        _type_: _description_
    """
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                            padding='same', activation='relu')(A_prev)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same', activation='relu')(conv1)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                            padding='same', activation='relu')(conv2)

    output = K.layers.add([conv3, A_prev])
    return K.activations.relu(output)