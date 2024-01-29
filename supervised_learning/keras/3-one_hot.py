#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """_summary_

    Args:
        labels ([type]): [description]
        classes ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    return K.utils.to_categorical(labels, classes)
