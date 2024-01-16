#!/usr/bin/env python3
"""_summary_"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """_summary_

    Args:
        alpha (_type_): _description_
        beta2 (_type_): _description_
        epsilon (_type_): _description_
        var (_type_): _description_
        grad (_type_): _description_
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (new_s ** (1 / 2) + epsilon)
    return var, new_s
