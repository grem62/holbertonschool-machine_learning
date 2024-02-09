#!/usr/bin/env python3
"""_summary_"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """_summary_

    Args:
        dA (_type_): _description_
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

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += \
                            np.multiply(mask, dA[i, h, w, c])
                    else:
                        da = dA[i, h, w, c]
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += \
                            np.ones(shape) * average

    return dA_prev
