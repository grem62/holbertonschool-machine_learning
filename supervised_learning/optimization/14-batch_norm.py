#!/usr/bin/env python3
"""_summary_"""


import tensorflow.compat.v1 as tf
"""_summary_"""


def create_batch_norm_layer(prev, n, activation):
    """_summary_"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n,
                                  activation=None,
                                  kernel_initializer=init,
                                  name="layer")
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    epsilon = 1e-8
    norm = tf.nn.batch_normalization(layer(prev), mean, variance,
                                     beta, gamma, epsilon)
    return activation(norm)
