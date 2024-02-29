#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """_summary_

    Args:
        X (type): _summary_
        nb_filters (type): _summary_
        growth_rate (type): _summary_
        layers (type): _summary_

    Returns:
        type: _summary_
    """
    init = K.initializers.he_normal()
    for i in range(layers):
        batch_norm1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(batch_norm1)
        conv1 = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                padding='same', kernel_initializer=init)(activation1)
        batch_norm2 = K.layers.BatchNormalization()(conv1)
        activation2 = K.layers.Activation('relu')(batch_norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                padding='same', kernel_initializer=init)(activation2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters