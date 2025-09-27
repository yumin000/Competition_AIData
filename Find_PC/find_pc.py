import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
# ... (다른 import)

print("--- Start Execution ---") # 1단계: 반드시 출력되어야 함

try:
    # 파일을 읽습니다.
    file_path = 'C:/Users/eunse/OneDrive/바탕 화면/hydro_sam/total_rename_data/trainData.csv'
    df = pd.read_csv(file_path, encoding='euc-kr')
   
    print("--- File Read Successful ---") # 2단계: 파일 읽기 성공 시 출력
except FileNotFoundError:
    print("ERROR: trainData.csv 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()
except Exception as e:
    # FileNotFoundError 이외의 다른 오류(예: encoding 문제 등)가 발생한 경우
    print(f"ERROR: 파일을 읽는 도중 예상치 못한 오류가 발생했습니다: {e}")
    exit()
    
# ... (이후 코드)

df=df.ffill()

group_pc_results=[]

i = 1

scaler=StandardScaler()
for code,group in df.groupby('code_new'):
   
    x=group.drop(['elev','ymd','code_new'],axis=1)
    e=group['elev'].values
    y=group['ymd'].values
    x=scaler.fit_transform(x)
    # PCA 수행
    pca = PCA(n_components=3) # 여기안에서 스케일링 한다는데
    pc_components = pca.fit_transform(x)

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
    #group_pc_results.append(PcDf)
    print(f"{i} Explained variance ration : ",pca.explained_variance_ratio_) #pc 각각의 분산 설명
    print(f"{i} Cumulative explained variance : ",np.cumsum(pca.explained_variance_ratio_)) #sum
    X_reconstructed = pca.inverse_transform(pc_components)
    reconstruction_error = np.mean((x - X_reconstructed) ** 2)
    print(f"{i} Reconstruction error:", reconstruction_error)
    i=i+1



#total_pc_df=pd.concat(group_pc_results,ignore_index=True)
#total_pc_df.to_csv('Find_PC/pc3_train.csv',index=False,encoding="euc-kr")