#!/usr/bin/env python3
"""This module contains the function for
calculating the moving average of a data set
"""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a
    data set with bias correction.
    """
    # Initialize an empty list to store the moving averages
    moving_averages = []

    # Initialize the moving average
    v = 0

    # Iterate over the data
    for i in range(len(data)):
        # Calculate the weighted average of the current
        # data point and the previous moving average
        v = beta * v + (1 - beta) * data[i]

        # Apply bias correction by scaling the moving average
        # based on the number of data points
        v_corrected = v / (1 - beta ** (i + 1))

        # Append the corrected moving average to the list
        moving_averages.append(v_corrected)

    # Return the list of moving averages
    return moving_averages
