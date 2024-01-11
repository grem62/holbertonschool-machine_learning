#!/usr/bin/env python3
"""

"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """_summary_

    Args:
        x (_type_): _description_
        layer_sizes (list, optional): _description_. Defaults to [].
        activations (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            pred = create_layer(x, layer_sizes[0], activations[0])
        else:
            pred = create_layer(x, layer_sizes[i], activations[i])
        return pred
