#!/usr/bin/env python3
"""_summary_
"""

import tensorflow.compat.v1 as tf
"""_summary_
"""


def evaluate(X, Y, save_path):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_
        save_path (_type_): _description_
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        return y_pred, accuracy, loss
