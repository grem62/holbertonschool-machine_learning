#!/usr/bin/env python3
"""sumary"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """_summary_

    Args:
        A_prev (_type_): _description_
        filters (_type_): _description_
        s (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    F11, F3, F12 = filters

    initializer = K.initializers.HeNormal(seed=None)

    # Main path
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation('relu')(batch_normalization_1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu_2 = K.layers.Activation('relu')(batch_normalization_2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_2)
    batch_normalization_3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shortcut connection
    shortcut_connection = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_shortcut = K.layers.BatchNormalization(
        axis=3)(shortcut_connection)

    # Add shortcut value to the output
    sum_result = K.layers.Add()(
        [batch_normalization_3, batch_normalization_shortcut])

    # Activate the final output
    activated_output = K.layers.Activation(activation='relu')(sum_result)

    return activated_output
