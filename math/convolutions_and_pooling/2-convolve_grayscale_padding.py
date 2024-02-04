#!/usr/bin/env python3
"""_summary_"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    Args:
        images (np.ndarray): shape (m, h, w) containing multiple grayscale
                             images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel (np.ndarray): shape (kh, kw) containing the kernel for the
                             convolution
            kh: height of the kernel
            kw: width of the kernel
        padding (tuple): (ph, pw)
            ph: padding for the height of the images
            pw: padding for the width of the images

    Returns:
        np.ndarray: The convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h - kh + 2 * ph + 1
    output_w = w - kw + 2 * pw + 1

    output = np.zeros((m, output_h, output_w))

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images_padded[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))

    return output
