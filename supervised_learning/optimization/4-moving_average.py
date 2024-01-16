#!/usr/bin/env python3
"""_summary_"""

import numpy as np
"""_summary_"""


def moving_average(data, beta):
    """_summary_

    Args:
        data (_type_): _description_
        beta (_type_): _description_
    """
    v = 0
    avg = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        avg.append(v / (1 - beta**(i + 1)))
    return avg
