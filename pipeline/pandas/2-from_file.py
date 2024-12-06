#!/usr/bin/env python3
"""This module loads data from a file into a dataframe"""
import pandas as pd


def from_file(filename, delimiter):
    """This function loads data from a file into a dataframe
    """
    return pd.read_csv(filename, delimiter=delimiter)
