import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random

# 데이터 로드
df = pd.read_csv('TST_test1.csv')
ddf=pd.read_csv('TST_test11.csv')

# 날짜
dates_arr = pd.to_datetime(df['ymd'])

# 플롯

for i in range(1, 13):
    plt.figure(figsize=(15, 7))
    col_name = str(i)  # ✅ 숫자를 문자열로 변환
    if col_name in df.columns:  # ✅ 실제 열 존재 확인
        plt.plot(dates_arr, df[col_name], '-', label=f"elev_{i}")
        plt.plot(dates_arr, ddf[col_name], '-', label=f"input_zero_elev_{i}")
    plt.xlabel("Date")
    plt.ylabel("Elev")
    plt.title(f"code {i}_Elev Prediction")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"TST/TST_test/show_graph/TST_predict_{i}.png")


