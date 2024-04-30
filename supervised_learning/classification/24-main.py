#!/usr/bin/env python3

import numpy as np
import os

oh_encode = __import__('24-one_hot_encode').one_hot_encode

lib = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)
