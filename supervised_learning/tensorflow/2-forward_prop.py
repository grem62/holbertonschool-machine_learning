#!/usr/bin/env python3
"""
_summary_
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer
"""_summary_
"""


def forward_prop(x, layer_sizes=[], activations=[]):
    """_summary_

    Args:
        x (_type_): _description_
        layer_sizes (_type_): _description_
        activations (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer

    return layer
