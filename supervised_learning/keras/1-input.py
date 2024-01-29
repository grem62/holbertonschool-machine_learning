#!/usr/bin/env python3
"""This module contains a function to build a Keras model.

The build_model function takes in the following parameters:
- nx: an integer representing the number of input features
- layers: a list of integers representing the number of nodes in each layer
- activations: a list of strings representing
the activation function for each layer
- lambtha: a float representing the L2 regularization parameter
- keep_prob: a float representing the probability of keeping
a node during dropout

The function returns a Keras model with the specified architecture.

Example usage:
model = build_model(10, [64, 32, 16], ['relu', 'relu', 'sigmoid'], 0.01, 0.8)
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a Keras model with the specified architecture.

    Args:
        nx (int): Number of input features.
        layers (list): Number of nodes in each layer.
        activations (list): Activation function for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability of keeping a node during dropout.

    Returns:
        Keras model: The built Keras model.
    """

    # Create the input layer
    model_input = K.Input(shape=(nx,))

    # Create the regularizer
    regularizer = K.regularizers.l2(lambtha)

    # Set the initial value of x to the model input
    x = model_input

    # Loop through the layers
    for i in range(len(layers)):
        # Create a dense layer with the specified
        # number of nodes, activation function, and regularizer
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=regularizer)(x)

        # Apply dropout if it's not the last layer
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Create the model with the input and output
    model = K.Model(inputs=model_input, outputs=x)

    return model
