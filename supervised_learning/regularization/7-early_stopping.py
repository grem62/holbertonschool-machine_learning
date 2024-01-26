#!/usr/bin/env python3
"""_summary_
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    early_stopping = True
    # Check if the current cost improvement is greater than the threshold
    if (opt_cost - cost) > threshold:
        # Reset the count as there is improvement
        count = 0
    else:
        # Increment the count as there is no significant improvement
        count += 1
    # Check if the count has not reached the patience limit
    if count != patience:
        # Continue training, no early stopping
        early_stopping = False

    # Stop training, early stopping
    return early_stopping, count
