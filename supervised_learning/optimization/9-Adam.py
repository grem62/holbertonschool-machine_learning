#!/usr/bin/env python3
"""_summary_"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """_summary_

    Args:
        alpha (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
        epsilon (_type_): _description_
        var (_type_): _description_
        grad (_type_): _description_
        v (_type_): _description_
        s (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    V1 = (beta1 * v) + ((1 - beta1) * grad)
    S1 = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    V2 = V1 / (1 - (beta1 ** t))
    S2 = S1 / (1 - (beta2 ** t))
    W = var - (alpha * (V2 / ((S2 ** (1/2)) + epsilon)))
    return W, V2, S2
