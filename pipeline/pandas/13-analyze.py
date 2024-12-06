#!/usr/bin/env python3
"""
This module analyzes data in a DataFrame
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp
    column.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df.describe()
