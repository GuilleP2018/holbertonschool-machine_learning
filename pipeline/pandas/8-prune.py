#!/usr/bin/env python3
"""This modlue prunes a dataframe to only contain the columns"""


def prune(df):
    """This function prunes nan fron close in a dataframe
        """
    return df.dropna(subset=['Close'])
