#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""_summary_
"""


def one_hot_encode(Y, classes):
    """_summary_

    Args:
        Y (_type_): _description_
        classes (_type_): _description_
    """
    try:
        encode = np.zeros((classes, Y.shape[0]))
        encode[Y, np.arange(Y.shape[0])] = 1
        return encode
    except Exception:
        return None
