#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum with tensorflow
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """_summary_

    Args:
        loss (_type_): _description_
        alpha (_type_): _description_
        beta1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)



    




