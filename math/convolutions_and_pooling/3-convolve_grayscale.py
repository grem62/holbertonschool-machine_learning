#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """_summary_

    Args:
        images (np.array): shape (m, h, w) containing multiple grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel (np.array): shape (kh, kw) containing the
        kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding (str, optional): 'same' or 'valid'
        stride (tuple, optional): (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image

    Returns:
        np.array: the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    m, h, w = images.shape

    h_conv = int((h - kh) / sh + 1)
    w_conv = int((w - kw) / sw + 1)
    convoluted = np.zeros((m, h_conv, w_conv))

    for i in range(h_conv):
        for j in range(w_conv):
            convoluted[:, i, j] = np.sum(
                images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return convoluted
