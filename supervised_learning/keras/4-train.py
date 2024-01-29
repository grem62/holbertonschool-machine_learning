#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, verbose=True, shuffle=False):
    """_summary_

    Args:
        network (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
        batch_size (_type_): _description_
        epochs (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.
        shuffle (bool, optional): _description_. Defaults to False.
    """
    model = K.Sequential(network)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, labels,
              batch_size=batch_size,
              epochs=epochs, verbose=verbose, shuffle=shuffle)
    return model
