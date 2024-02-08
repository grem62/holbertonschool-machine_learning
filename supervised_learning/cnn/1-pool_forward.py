#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """_summary_

    Args:
        A_prev (_type_): _description_
        kernel_shape (_type_): _description_
        stride (tuple, optional): _description_. Defaults to (1, 1).
        mode (str, optional): _description_. Defaults to 'max'.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    A_new = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            A_new[:, i, j, :] = np.max(
                A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                axis=(1, 2)
            ) if mode == 'max' else np.average(
                A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                axis=(1, 2)
            )

    return A_new
