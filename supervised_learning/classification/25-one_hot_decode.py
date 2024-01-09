#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""_summary_
"""


def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
