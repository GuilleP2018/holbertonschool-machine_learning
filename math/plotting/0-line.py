#!/usr/bin/env python3
"""Line plot"""
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Line plot"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')  # 'r-' specifies a red line
    plt.xlim(0, 10)  # Set the x-axis range from 0 to 10
    plt.show()

