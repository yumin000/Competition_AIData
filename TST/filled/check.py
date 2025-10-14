import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# CSV 파일 로드 (가정: 첫 번째 열은 시간/날짜 정보, 나머지 열은 시계열 데이터)
# 실제 파일 경로에 맞게 수정하세요
file_path = 'TST_test5_.csv'
df = pd.read_csv(file_path)

# 컬럼 이름 지정 (예시)
# 첫 번째 컬럼은 'ymd', 나머지는 '1', '2', ... 라고 가정
df.columns = ['ymd'] + [str(i) for i in range(1, 13)] 

# 'ymd' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정
df['ymd'] = pd.to_datetime(df['ymd'])
df = df.set_index('ymd')

# 결측치를 NaN으로 변환 (CSV 로드 시 문자열 등으로 인식될 경우 대비)
df = df.replace('결측값표시', np.nan) # '결측값표시'는 실제 파일의 결측치 표현 방식에 따라 수정

# 데이터 타입 확인 및 숫자로 변환 (필요한 경우)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("결측치 개수 (열별):")
print(df.isnull().sum())