#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """_summary_

    Args:
        Observation (_type_): _description_
        Emission (_type_): _description_
        Transition (_type_): _description_
        Initial (_type_): _description_

    Returns:
        _type_: _description_
    """
    Ob = Observation.shape[0]
    N = Transition.shape[0]
    V = np.zeros((N, Ob))
    B = np.zeros((N, Ob))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, Ob):
        for j in range(N):
            V[j, t] = np.max(V[:, t - 1] *
                             Transition[:, j] * Emission[j, Observation[t]])
            B[j, t] = np.argmax(V[:, t - 1] *
                                Transition[:, j] * Emission[j, Observation[t]])
    P = np.max(V[:, Ob - 1])
    path = [np.argmax(V[:, Ob - 1])]
    for t in range(Ob - 1, 0, -1):
        path.insert(0, int(B[path[0], t]))
    return path, P
