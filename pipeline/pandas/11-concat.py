#!/usr/bin/env python3
"""This module concatenates two dataframes"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """This function concatenates two dataframes
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
