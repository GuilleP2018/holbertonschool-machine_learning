#!/usr/bin/env python3
"""This modlue creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """This function creates a pd.DataFrame from a np.ndarray
    """
    # Generate the columns
    columns = [chr(i) for i in range(65, 91)]

    return pd.DataFrame(array, columns=columns[:array.shape[1]])
