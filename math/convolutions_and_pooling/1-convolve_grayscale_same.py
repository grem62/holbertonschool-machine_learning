#!/usr/bin/env python3
"""_summary_"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images

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

    Returns:
        np.ndarray: The convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    output_h = h
    output_w = w

    output = np.zeros((m, output_h, output_w))

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images[:, i: i + kh, j: j + kw] * kernel
                               ).sum(axis=(1, 2))

    return output
