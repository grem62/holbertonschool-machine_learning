#!/usr/bin/env python3
"""_summary_"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """_summary_

    Args:
        network (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    return network.evaluate(data, labels, verbose=verbose)
