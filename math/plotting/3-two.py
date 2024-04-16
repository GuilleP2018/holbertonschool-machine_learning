#!/usr/bin/env python3
"""Generate a plot x->y1 and x->y2"""
import numpy as np
import matplotlib.pyplot as plt

def two():
    """Generate a plot x->y1 and x->y2"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plotting
    plt.plot(x, y1, 'r--', label='C-14')  # 'r--' specifies a dashed red line
    plt.plot(x, y2, 'g-', label='Ra-226')  # 'g-' specifies a solid green line

    # Labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Axis range
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Legend
    plt.legend(loc='upper right')

    plt.show()