import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random
import os

# 데이터 로드
df = pd.read_csv('TST_test1.csv')
ddf = pd.read_csv('TST_test11.csv')

# 날짜
dates_arr = pd.to_datetime(df['ymd'])
min_len = min(len(df), len(ddf))

dates_arr = pd.to_datetime(df['ymd'])[:min_len]

for i in range(1, 13):
    plt.figure(figsize=(15, 7))
    col_name = str(i)
    if col_name in df.columns and col_name in ddf.columns:
        plt.plot(dates_arr, df[col_name][:min_len], '-', label=f"elev_{i}")
        plt.plot(dates_arr, ddf[col_name][:min_len], '-', label=f"input_zero_elev_{i}")
    plt.xlabel("Date")
    plt.ylabel("Elev")
    plt.title(f"code {i}_Elev Prediction")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"TST/TST_test/show_graph/TST_predict_{i}.png")
    plt.close()




