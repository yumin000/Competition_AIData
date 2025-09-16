import pandas as pd
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 경고 메시지 무시

import datetime as datetime

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')
df['ymd'] = pd.to_datetime(df['ymd'])

# 'ymd' 컬럼을 df의 인덱스로 설정
df.set_index('ymd', inplace=True)

pilot_code = 6
pilot_df = df[df['code_new'] == pilot_code].copy()

# 일별 데이터를 월별 데이터로 변환
monthly_df = pilot_df.resample('M').mean()

# 결측치는 보간으로 채움
if pilot_df.isnull().sum().sum() > 0:
    pilot_df.interpolate(method='time', inplace=True)

endog_var = pilot_df['elev'] # 목표 변수
exog_vars = pilot_df[['temp', 'gtemp', 'pressure']] # 외부 변수

# 일반 차수 (p, d, q)는 이전에 찾은 (1, 0, 0) 사용 or 간단하게 고정
p=1; d=1; q=0

# 최적의 계절성 차수 (P, D, Q, s) 찾기 - AIC 기준
P_range = range(0, 3)
D_range = [1]
Q_range = range(0, 3)
s=12

seasonal_pdq_combinations = list(itertools.product(P_range, D_range, Q_range))

best_aic = float('inf')
best_seasonal_order = None

print(f"\n{datetime.datetime.now()}")
print("\n최적의 계절성 SARIMAX 차수(P, D, Q)를 찾는 중 ...")
for seasonal_order in seasonal_pdq_combinations:
    try:
        model = sm.tsa.SARIMAX(endog=endog_var,
                               exog=exog_vars,
                               order=(p, d, q),
                               seasonal_order=seasonal_order + (s,))
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best_seasonal_order = seasonal_order
    except Exception as e:
        continue

print(f"\n최적의 계절성 차수 (P, D, Q) : {best_seasonal_order} (AIC : {best_aic: .2f})")
print(f"\n{datetime.datetime.now()}")