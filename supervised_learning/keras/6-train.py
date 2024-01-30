#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def train_model(network, data,
                labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """_summary_

    Args:
        network (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
        batch_size (_type_): _description_
        epochs (_type_): _description_
        validation_data (_type_, optional): _description_. Defaults to None.
        early_stopping (bool, optional): _description_. Defaults to False.
        patience (int, optional): _description_. Defaults to 0.
        verbose (bool, optional): _description_. Defaults to True.
        shuffle (bool, optional): _description_. Defaults to False.
    """

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)
    return history
