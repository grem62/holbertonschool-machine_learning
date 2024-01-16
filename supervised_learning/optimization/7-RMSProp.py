#!/usr/bin/env python3
"""_summary_"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    new_s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (new_s ** (1 / 2) + epsilon)
    return var, new_s
