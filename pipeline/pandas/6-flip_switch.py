#!/usr/bin/env python3
"""This module sorts data in reverse chronological order"""


def flip_switch(df):
    """This function sorts data in reverse chronological
    order
    """
    return df.sort_index(ascending=False).transpose()
