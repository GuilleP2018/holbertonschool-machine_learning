#!/usr/bin/env python3
"""This module slices a dataframe along the
columns High and Close"""


def slice(df):
    """This function slices a dataframe along the
    columns High and Close.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
