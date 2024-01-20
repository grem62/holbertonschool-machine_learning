#!/usr/bin/env python3
""" Module that creates a neural network. """

import tensorflow.compat.v1 as tf
import numpy as np
""" model function """

def create_batch(data, batch_size):
    """ Function that creates batches from a data set. """
    num_batches = len(data) // batch_size
    if len(data) % batch_size != 0:
        num_batches += 1
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = data[start:end]
        batches.append(batch)
    return batches

def model(Data_train, Data_valid, layers, 
          activations, alpha=0.001, beta1=0.9, 
          beta2=0.999, epsilon=1e-8, decay_rate=1, 
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ Function that creates a neural network. """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, nx = X_train.shape
    classes = Y_train.shape[1]

    tf.set_random_seed(0)
    np.random.seed(0)

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    for i in range(len(layers)):
        if i == 0:
            prev = x
        prev = tf.layers.Dense(units=layers[i], activation=activations[i], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))(prev)
        if i < len(layers) - 1:
            prev = tf.layers.BatchNormalization()(prev)

    y_pred = prev

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))

    global_step = tf.Variable(0, trainable=False)
    decay_steps = m // batch_size
    learning_rate = tf.train.inverse_time_decay(alpha, global_step, decay_steps, decay_rate, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)
            batches = create_batch(list(zip(X_train_shuffled, Y_train_shuffled)), batch_size)

            train_cost = 0
            train_accuracy = 0
            valid_cost = 0
            valid_accuracy = 0

            for step, batch in enumerate(batches, 1):
                batch_X, batch_Y = zip(*batch)
                _, batch_cost, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: batch_X, y: batch_Y})

                train_cost += batch_cost
                train_accuracy += batch_accuracy

                if step % 100 == 0 or step == len(batches):
                    print("Step {}: Cost = {:.4f}, Accuracy = {:.4f}".format(step, batch_cost, batch_accuracy))

            train_cost /= len(batches)
            train_accuracy /= len(batches)

            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {:.4f}".format(train_cost))
            print("\tTraining Accuracy: {:.4f}".format(train_accuracy))
            print("\tValidation Cost: {:.4f}".format(valid_cost))
            print("\tValidation Accuracy: {:.4f}".format(valid_accuracy))

        saver.save(sess, save_path)
        print("Model saved in path: {}".format(save_path))

    return save_path

