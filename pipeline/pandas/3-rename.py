#!/usr/bin/env python3
"""This module renames a column of a dataframe"""
import pandas as pd


def rename(df):
    """This function renames the column Timestamp to Datetime,
    converts the timestamp values to datetime values,
    and displays only the Datetime and Close columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
