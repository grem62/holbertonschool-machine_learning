#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network with the Keras library.

    Arguments:
    nx -- number of input features
    layers -- list containing the number of nodes in each layer of the network
    activations -- list containing the activation functions used for each layer
    lambtha -- L2 regularization parameter
    keep_prob -- probability that a node will be kept for dropout

    Returns:
    model -- a Keras model instance
    """

    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    # Add first densely connected layer with input shape
    model.add(K.layers.Dense(layers[0],
                             activation=activations[0], input_shape=(nx,),
                             kernel_regularizer=regularizer))

    for i in range(1, len(layers)):
        # Add dropout layer
        model.add(K.layers.Dropout(1 - keep_prob))
        # Add densely connected layer
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=regularizer))

    return model
