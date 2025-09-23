import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv('total_rename_data/trainData.csv')
df=df.fillna(method='ffill')

group_pc_results=[]

for code,group in df.groupby('code_new'):
    x=group.drop(['elev','ymd','code_new'],axis=1)
    e=group['elev'].values
    y=group['ymd'].values
 # 데이터 스케일링
    x_scaled = StandardScaler().fit_transform(x)

    # PCA 수행
    pca = PCA(n_components=3)
    pc_components = pca.fit_transform(x_scaled)

    # PCA 결과를 DataFrame으로 변환
    # 이 부분에서 열 이름을 명확하게 지정하는 것이 중요합니다.
    PcDf = pd.DataFrame(data=pc_components, columns=['pc1', 'pc2', 'pc3'])

    # 기존 열('code_new', 'ymd', 'elev') 추가
    PcDf['code_new'] = code
    PcDf['ymd'] = group['ymd'].values
    PcDf['elev'] = group['elev'].values

    # 열 순서 재배치
    PcDf = PcDf[['code_new', 'ymd', 'pc1', 'pc2', 'pc3', 'elev']]

    # 그룹별 결과를 리스트에 추가
    group_pc_results.append(PcDf)



total_pc_df=pd.concat(group_pc_results,ignore_index=True)
total_pc_df.to_csv('Find_PC/pc3_grouped_code.csv',index=False,encoding="euc-kr")