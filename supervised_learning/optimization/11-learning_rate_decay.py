#!/usr/bin/env python3
"""_summary_"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """"_summary_"""
    alpha = decay_rate * (1 - (global_step // decay_step))
    return alpha
