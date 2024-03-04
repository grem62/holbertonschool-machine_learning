#!/usr/bin/env python3
"""_summary_
"""

import tensorflow as tf


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
    X = tf.keras.layers.BatchNormalization()(X)

    # ReLU activation
    X = tf.keras.layers.Activation('relu')(X)

    # Convolutional layer
    X = tf.keras.layers.Conv2D(int(nb_filters * compression),
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer='he_normal')(X)

    # Average pooling
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, int(nb_filters * compression)
