#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""_summary_
"""


def one_hot_decode(one_hot):
    """_summary_

    Args:
        one_hot (_type_): _description_
    """
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
