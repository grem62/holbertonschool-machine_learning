#!/usr/bin/env python3
"""_summary_

"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """_summary_

    Args:
        A_prev (_type_): _description_
        filters (_type_): _description_
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                            strides=(s, s), padding='same',
                            activation='relu')(A_prev)
    normalisation1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation('relu')(normalisation1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same', activation='relu')(relu_1)
    normalisation2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(normalisation2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                            padding='same', activation='relu')(relu2)
    normalisation3 = K.layers.BatchNormalization(axis=3)(conv3)
    relu3 = K.layers.Activation('relu')(normalisation3)
    conv4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                            strides=(s, s), padding='same',
                            activation='relu')(A_prev)
    normalisation4 = K.layers.BatchNormalisation(axis=3)(conv4)
    relu4 = K.layers.Activation('relu')(normalisation4)
    output = K.layers.add([normalisation3, normalisation4])
    return K.activations.relu(output)
