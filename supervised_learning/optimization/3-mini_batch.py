#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
"""summary"""


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
        epochs (int, optional): _description_. Defaults to 5
        load_path (str, optional): _description_. Defaults to
        "/tmp/model.ckpt".
        save_path (str, optional): _description_. Defaults to
        "/tmp/model.ckpt".

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
                                            feed_dict={x: X_train,
                                                       y: Y_train})
            validation_cost = sess.run(loss,
                                          feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = sess.run(accuracy,
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {training_cost}")
            print(f"\tTraining Accuracy: {training_accuracy}")
            print(f"\tValidation Cost: {validation_cost}")
            print(f"\tValidation Accuracy: {validation_accuracy}")


            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if i % 100 == 0 and i != 1:
                    step_number = i // batch_size
                    step_cost, step_accuracy = sess.run([loss, accuracy],
                                                        feed_dict={x: X_batch,
                                                                   y: Y_batch})
                    print(f"Step {step_number}:")
                    print(f"\tCost: {step_cost}")
                    print(f"\tAccuracy: {step_accuracy}")


        saver.save(sess, save_path)
        return save_path
