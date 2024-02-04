#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """_summary_

    Args:
        images (np.array): shape (m, h, w, c) containing
        multiple grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the image
        kernel (np.array): shape (kh, kw, c) containing the
        kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding (str, optional): indicates the type of padding
            'same': same padding
            'valid': valid padding
            Default: 'same'
        stride (tuple, optional): shape (sh, sw) containing
        the strides for the convolution
            sh: stride for the height
            sw: stride for the width
            Default: (1, 1)

    Returns:
        np.array: the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), 'constant')

    ch = int((h + 2 * ph - kh) / sh + 1)
    cw = int((w + 2 * pw - kw) / sw + 1)

    convoluted = np.zeros((m, ch, cw, nc))

    for i in range(ch):
        for j in range(cw):
            for k in range(nc):
                images_slide = images_padded[:, i * sh:i * sh + kh, j *
                                             sw:j * sw + kw, :]
                convoluted[:, i, j, k] = np.sum(images_slide *
                                                kernels[:, :, :, k],
                                                axis=(1, 2, 3))

    return convoluted
