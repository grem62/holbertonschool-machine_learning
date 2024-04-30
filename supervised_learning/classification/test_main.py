#!/usr/bin/env python3

import numpy as np
Neuron = __import__('0-neuron').Neuron

try:
    nn = Neuron(0)
    print('FAIL')
except ValueError as e:
    if str(e) == 'nx must be a positive integer':
        print('OK', end='')
    else:
        print('FAIL')
