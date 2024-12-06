#!/usr/bin/env python3
"""This modulle visualizes the data from 2017-2019"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df: pd.DataFrame = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])

df = df.rename(columns={"Timestamp": "Date"})

df["Date"] = pd.to_datetime(df["Date"], unit='s')

df = df.set_index(["Date"])

df["Close"] = df["Close"].ffill()

for column in ["High", "Low", "Open"]:
    if column in df.columns:
        df[column] = df[column].fillna(df["Close"])

for column in ["Volume_(BTC)", "Volume_(Currency)"]:
    if column in df.columns:
        df[column] = df[column].fillna(0)

df = df[df.index >= "2017-01-01"]


daily_df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

print(daily_df)

fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axs[0].plot(daily_df.index, daily_df["High"], label="High", color="green")
axs[0].plot(daily_df.index, daily_df["Low"], label="Low", color="red")
axs[0].plot(daily_df.index, daily_df["Open"], label="Open", color="orange")
axs[0].plot(daily_df.index, daily_df["Close"], label="Close", color="blue")
axs[0].set_title("Daily Prices (High, Low, Open, Close)", fontsize=16)
axs[0].set_ylabel("Price (USD)", fontsize=14)
axs[0].legend()
axs[0].grid()

axs[1].plot(daily_df.index, daily_df["Volume_(BTC)"],
            label="Volume (BTC)", color="purple")
axs[1].set_title("Daily Volume (BTC)", fontsize=16)
axs[1].set_ylabel("Volume (BTC)", fontsize=14)
axs[1].legend()
axs[1].grid()

axs[2].plot(daily_df.index, daily_df["Volume_(Currency)"],
            label="Volume (Currency)", color="brown")
axs[2].set_title("Daily Volume (Currency)", fontsize=16)
axs[2].set_ylabel("Volume (USD)", fontsize=14)
axs[2].set_xlabel("Date", fontsize=14)
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.savefig("Visualize_BTC_2017-2019.png")
plt.show()
