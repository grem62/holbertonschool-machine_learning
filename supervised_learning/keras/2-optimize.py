#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """_summary_

    Args:
        network (_type_): _description_
        alpha (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return None
