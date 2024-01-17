#!/usr/bin/env python3
"""_summary_"""

import tensorflow.compat.v1 as tf
"""_summary_"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """_summary_"""
    learning_rate_decay = tf.train.inverse_time_decay(learning_rate=alpha,
                                                      global_step=global_step,
                                                      decay_steps=decay_step,
                                                      decay_rate=decay_rate,
                                                      staircase=True)
    return learning_rate_decay
