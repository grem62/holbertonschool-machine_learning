#!/usr/bin/env python3
"""_summary_"""


import numpy as np
"""_summary_"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """_summary_"""
    update = beta1 * v + (1 - beta1) * grad
    new_var = var - alpha * update
    return new_var, update
