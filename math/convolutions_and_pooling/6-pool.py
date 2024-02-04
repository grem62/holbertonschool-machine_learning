#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """_summary_

    Args:
        images (np.array): shape (m, h, w, c) containing
        multiple grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the image
        kernel_shape (tuple): shape (kh, kw) containing the
        size of the kernel for the pooling
            kh: height of the kernel
            kw: width of the kernel
        stride (tuple): shape (sh, sw) containing the strides
        for the pooling
            sh: stride for the height
            sw: stride for the width
        mode (str, optional): indicates the type of pooling
            'max': max pooling
            'avg': average pooling
            Default: 'max'

    Returns:
        np.array: the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ch = int((h - kh) / sh + 1)
    cw = int((w - kw) / sw + 1)

    pooled = np.zeros((m, ch, cw, c))

    for i in range(ch):
        for j in range(cw):
            images_slide = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(images_slide, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.mean(images_slide, axis=(1, 2))

    return pooled
