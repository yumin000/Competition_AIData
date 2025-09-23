import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

df=pd.read_csv('Find_PC/pc3_grouped_code.csv')
df['ymd'] = pd.to_datetime(df['ymd'], errors='coerce')
for i in range(2,4):
    for code, group in df.groupby('code_new'):
        plt.figure(figsize=(20,6))
        plt.plot(group['ymd'].iloc[::50],group[f'pc{i}'].iloc[::50],label=str(code))
        year_ticks=group.groupby(group['ymd'].dt.year)['ymd'].first()
        plt.xticks(year_ticks,year_ticks.dt.strftime('%Y'))
        plt.xlabel('ymd')
        plt.ylabel(f'PC{i}')
        plt.title(f'ymd - {code}PC{i}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Find_PC/Graph_PC{i}/{code} PC{i}.png")
        plt.close()

