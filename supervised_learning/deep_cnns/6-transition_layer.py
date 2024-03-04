#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """_summary_

    Args:
        X (_type_): _description_
        nb_filters (_type_): _description_
        compression (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Batch Normalization
    X = K.layers.BatchNormalization()(X)

    # ReLU activation
    X = K.layers.Activation('relu')(X)

    # Convolutional layer
    X = K.layers.Conv2D(int(nb_filters * compression), kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)

    # Average pooling
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, int(nb_filters * compression)
