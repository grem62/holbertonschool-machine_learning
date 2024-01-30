#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def save_model(network, filename):
    """_summary_

    Args:
        network (_type_): _description_
        filename (_type_): _description_
    """
    network.save(filename)
    return None


def load_model(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    return K.models.load_model(filename)
