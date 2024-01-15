#!/usr/bin/env python3
"""_summary_"""


import numpy as np
"""_summary_
"""


def normalize(X, m, s):
    """_summary_

    Args:
        X (_type_): _description_
        m (_type_): _description_
        s (_type_): _description_
    """
    normal = (X - m) / s
    return normal
