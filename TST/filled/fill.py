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



# --- 1. 데이터 준비 (이 부분은 이전과 동일하게 유지) ---
# ... (파일 로드, 인덱스 설정, NaN 변환 과정) ...

# 결측치를 저장할 새로운 데이터프레임
df_imputed = df.copy()

# SARIMA 모델 파라미터 (p, d, q)와 계절성 파라미터 (P, D, Q, s) 설정
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12) # 예시: 월별 계절성 가정

for col in df.columns:
    print(f"\n--- 컬럼 '{col}' 결측치 보간 시작 ---")
    
    # 해당 열의 시계열 데이터
    ts = df[col]
    
    missing_indices = ts[ts.isnull()].index
    
    if len(missing_indices) == 0:
        print(f"컬럼 '{col}': 결측치가 없습니다. 건너뜜.")
        continue

    try:
        # **수정된 핵심 부분:**
        # 1. 결측치가 있는 원본 시계열(ts) 전체를 endog(종속변수)로 사용합니다.
        #    SARIMAX 모델이 내부적으로 NaN 값을 처리하여 보간합니다.
        model = SARIMAX(
            ts, # ts_clean 대신 결측치가 포함된 ts 사용
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        # model.fit()을 호출하면 NaN 위치의 값이 예측값으로 채워집니다.
        results = model.fit(disp=False)
        
        print(f"컬럼 '{col}': SARIMA 모델 적합 완료")
        
        # 2. 모델 적합 후, get_prediction()을 사용하여 적합된 값(in-sample)을 가져옵니다.
        #    이 적합값에는 NaN이었던 부분에 대한 보간 값이 포함되어 있습니다.
        #    start와 end는 훈련 데이터의 전체 인덱스를 사용합니다.
        
        # 전체 훈련 기간에 대한 예측값 (보간 값 포함)
        imputed_values = results.predict(start=ts.index.min(), end=ts.index.max(), dynamic=False)
        
        # 3. 원본 데이터프레임의 NaN 위치를 보간 값으로 대체합니다.
        df_imputed.loc[missing_indices, col] = imputed_values.loc[missing_indices]

        print(f"컬럼 '{col}': {len(missing_indices)}개의 결측치 SARIMA 보간 완료")

    except Exception as e:
        print(f"컬럼 '{col}': SARIMA 모델 적합/보간 중 오류 발생: {e}")
        # 오류 발생 시 대체 로직
        df_imputed.loc[missing_indices, col] = df[col].interpolate(method='time').loc[missing_indices]
        print(f"컬럼 '{col}': 대신 'time' 기반 선형 보간을 사용했습니다.")

# 최종 결측치 확인
print("\n--- 보간 후 최종 결측치 개수 (열별) ---")
print(df_imputed.isnull().sum())


df_imputed.to_csv("filled_Sarima_TST_test_5.csv")