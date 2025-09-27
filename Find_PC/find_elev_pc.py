import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ... (다른 import)



file_path = 'C:/Users/eunse/OneDrive/바탕 화면/hydro_sam/total_rename_data/trainData.csv'
df = pd.read_csv(file_path, encoding='euc-kr')
   
x=df[['elev']]
scaler = StandardScaler(feature_range=(-10, 30))
X_scaled = scaler.fit_transform(x)

df['elev']=X_scaled
df['ymd'] = pd.to_datetime(df['ymd'])

for code,group in df.groupby('code_new'):
    plt.figure(figsize=(20,6))
    plt.plot(group['ymd'].iloc[::50],group['elev'].iloc[::50],label=str(code))
    year_ticks=group.groupby(group['ymd'].dt.year)['ymd'].first()
    plt.xticks(year_ticks,year_ticks.dt.strftime('%Y'))
    plt.xlabel('ymd')
    plt.ylabel(f'elev')
    plt.title(f'code new :{code} [ elev ]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Find_PC/Graph_{code}_elev.png")
    plt.close()



