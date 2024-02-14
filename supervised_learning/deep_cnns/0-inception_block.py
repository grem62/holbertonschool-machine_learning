#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """_summary_

    Args:
        A_prev (_type_): _description_
        filters (_type_): _description_

    Returns:
        _type_: _description_
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    # 1x1 convolution before 3x3 convolution branch
    conv3x3_reduce = K.layers.Conv2D(filters=F3R,
                                     kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')(conv3x3_reduce)

    # 1x1 convolution before 5x5 convolution branch
    conv5x5_reduce = K.layers.Conv2D(filters=F5R,
                                     kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    # Max pooling branch
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(1, 1), padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu')(max_pool)

    # Concatenate all branches
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, conv_pool],
                                  axis=3)

    return output
