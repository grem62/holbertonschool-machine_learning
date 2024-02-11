#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)

    # Max pooling layer 1
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)

    # Max pooling layer 2
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    # Flatten the previous layer
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(units=120,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)(flatten)

    # Fully connected layer 2
    fc2 = tf.layers.Dense(units=84,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)(fc1)

    # Output layer
    output = tf.layers.Dense(units=10,
                             activation=tf.nn.softmax,
                             kernel_initializer=initializer)(fc2)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)

    # Accuracy
    accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accu
