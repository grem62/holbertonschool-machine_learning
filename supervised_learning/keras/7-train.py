#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
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
        learning_rate_decay (bool, optional): _description_. Defaults to False.
        alpha (float, optional): _description_. Defaults to 0.1.
        decay_rate (int, optional): _description_. Defaults to 1.
        verbose (bool, optional): _description_. Defaults to True.
        shuffle (bool, optional): _description_. Defaults to False.
    """

    def lr_schedule(epoch):
        """Fonction pour la décroissance du taux d'apprentissage"""
        return alpha / (1 + epoch * decay_rate)

    callbacks = []

    if learning_rate_decay and validation_data:
        lr_decay = K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_decay)

    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               verbose=verbose,
                                               mode='min')
        callbacks.append(early_stop)

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          shuffle=shuffle, validation_data=validation_data,
                          callbacks=callbacks)
    return history
