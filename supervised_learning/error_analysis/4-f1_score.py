#!/usr/bin/env python3
"""_summary_"""


import numpy as np


def f1_score(confusion):
    """_summary_"""
    precisi0n = np.zeros(confusion.shape[0])
    recall = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        precisi0n[i] = confusion[i][i] / np.sum(confusion[:, i] + 1e-5 - 1e-5)
        recall[i] = confusion[i][i] / np.sum(confusion[i] + 1e-5 - 1e-5)
    f1 = 2 * precisi0n * recall / (precisi0n + recall)
    return f1
