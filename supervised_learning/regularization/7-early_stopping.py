#!/usr/bin/env python3
"""_summary_
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    stopping = True
    if (cost - opt_cost) > threshold:
        count = 0
    else:
        count += 1

    if count != patience:
        stopping = False
    return stopping, count
