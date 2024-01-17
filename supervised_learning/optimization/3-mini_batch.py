#!/usr/bin/env python3
"""_summary_"""

import tensorflow.compat.v1 as tf
"""_summary_

    Returns:
        _type_: _description_
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data
"""
_summary_
"""


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """_summary_

    Args:
        X_train (_type_): _description_
        Y_train (_type_): _description_
        X_valid (_type_): _description_
        Y_valid (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        epochs (int, optional): _description_. Defaults to 5.
        load_path (str, optional): _description_.
        Defaults to "/tmp/model.ckpt".
        save_path (str, optional): _description_.
        Defaults to "/tmp/model.ckpt".

    Returns:
        _type_: _description_
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            # Evaluate the training cost and accuracy of the model
            training_cost = sess.run(loss,
                                     feed_dict={x: X_train, y: Y_train})
            training_accuracy = sess.run(accuracy,
                                         feed_dict={x: X_train, y: Y_train})
            validation_cost = sess.run(loss,
                                       feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = sess.run(accuracy,
                                           feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {training_cost}")
            print(f"\tTraining Accuracy: {training_accuracy}")
            print(f"\tValidation Cost: {validation_cost}")
            print(f"\tValidation Accuracy: {validation_accuracy}")

            if epoch == epochs:
                saver.save(sess, save_path)
                return save_path

            # shuffle the training data for each epoch
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            # Generate indices for mini-batch
            steps = list(range(0, len(shuffled_X), batch_size))

            # Loop over mini-batch
            for step, start_index in enumerate(steps, start=1):
                end_index = start_index + batch_size
                X_batch = shuffled_X[start_index:end_index]
                Y_batch = shuffled_Y[start_index:end_index]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print progress every 100 steps
                if step % 100 == 0 and step != 1:
                    step_cost, step_accuracy = sess.run([loss, accuracy],
                                                        feed_dict={x: X_batch,
                                                                   y: Y_batch})
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

        saver.save(sess, save_path)
        return save_path
