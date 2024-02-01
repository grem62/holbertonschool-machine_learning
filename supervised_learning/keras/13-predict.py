#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """_summary_

    Args:
        network (_type_): _description_
        data (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return network.predict(data, verbose=verbose)
