#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """_summary_

    Args:
        A_prev (_type_): _description_
        W (_type_): _description_
        b (_type_): _description_
        activation (_type_): _description_
        padding (str, optional): _description_. Defaults to "same".
        stride (tuple, optional): _description_. Defaults to (1, 1).
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    h_new = int((h_prev - kh + 2 * ph) / sh) + 1
    w_new = int((w_prev - kw + 2 * pw) / sw) + 1

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    A_new = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                A_new[:, i, j, k] = np.sum(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j
                           * sw + kw, :] * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return activation(A_new + b)
