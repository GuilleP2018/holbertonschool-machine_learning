#!/usr/bin/env python3
"""This module converts a dataframe to a numpy array"""


def array(df):
    """This function converts a dataframe to a numpy array.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
