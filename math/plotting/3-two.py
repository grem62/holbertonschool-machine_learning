#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Line graphs
plt.plot(x, y1, 'r--', label='C-14')  # Dashed red line for y1
plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green line for y2

# Labels and title
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')

# Legend in the upper right-hand corner
plt.legend(loc='upper right')

# Show the plot
plt.show()
