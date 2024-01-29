#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    model_input = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    x = model_input

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=model_input, outputs=x)
    return model
