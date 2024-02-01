#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def save_config(network, filename):
    """_summary_

    Args:
        network (_type_): _description_
        filename (_type_): _description_
    """
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    return K.models.model_from_json(filename)
