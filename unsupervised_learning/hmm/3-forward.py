import numpy as np


def forward(Observation, Emission, Transition, Initial):
    Ob = Observation.shape[0]
    N = Transition.shape[0]
    F = np.zeros((N, Ob))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, Ob):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] *
                             Transition[:, j] * Emission[j, Observation[t]])
    P = np.sum(F[:, Ob - 1])
    return P, F
