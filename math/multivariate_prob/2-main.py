#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

try:
    MultiNormal(np.array([[1], [2], [3], [4]]))
except ValueError as e:
    print(str(e))
    