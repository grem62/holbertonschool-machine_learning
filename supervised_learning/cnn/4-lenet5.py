#!/usr/bin/env python3

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """_summary_

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation=activation, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=activation,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = tf.layers.Flatten()(pool2)
    full1 = tf.layers.Dense(units=120, activation=activation,
                            kernel_initializer=init)(flatten)
    full2 = tf.layers.Dense(units=84, activation=activation,
                            kernel_initializer=init)(full1)
    y_pred = tf.layers.Dense(units=10, kernel_initializer=init)(full2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),
                                          tf.argmax(y_pred, 1)), tf.float32))
    return y_pred, train_op, loss, acc
