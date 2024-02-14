#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """_summary_

    Args:
        A_prev (_type_): _description_
        filters (_type_): _description_
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    chemin1 = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    chemin11 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                               padding='same', activation='relu')(chemin1)

    chemin2 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    chemin3 = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    chemin33 = K.layers.Conv2D(filters='F5', kernel_size=(5, 5),
                               padding='same', activation='relu')(chemin3)

    chemin4 = K.layers.MaxPooling2D(filters='filters', kernel_size=(3, 3),
                                    padding='same', aactivation='relu')(A_prev)

    chemin44 = K.layers.Conv2D(filters='FPP', kernel_size=(1, 1),
                               padding='same', activation='relu')(A_prev)

    concatenate = K.layers.concatenate([chemin1, chemin11, chemin2,
                                        chemin3, chemin33, chemin4,
                                        chemin44], axis='1')

    return concatenate
