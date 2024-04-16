#!/usr/bin/env python3
"""Generate a plot of a histogram"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Generate a plot of a histogram"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plotting
    bins = range(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    plt.yticks(np.arange(0, 31, 5))

    plt.show()
