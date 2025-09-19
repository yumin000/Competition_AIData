import pandas as pd
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 경고 메시지 무시

import numpy as np

# NSE 계산 함수
def cal_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator/denominator)
    return nse

# KGE 계산 함수
def cal_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1] # 상관계수
    beta = np.mean(y_pred)/np.mean(y_true) # 편향
    gamma = np.std(y_pred)/np.std(y_true) # 변동성
    kge = 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
    return kge 

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')
df['ymd'] = pd.to_datetime(df['ymd'])

# 'ymd' 컬럼을 df의 인덱스로 설정
df.set_index('ymd', inplace=True)

pilot_code = 12
pilot_df = df[df['code_new'] == pilot_code].copy()

# 월 별 데이터로 리샘플링
monthly_df = pilot_df.resample('M').mean()
if monthly_df.isnull().sum().sum() > 0:
    monthly_df.interpolate(method='time', implace=True)

endog_var = monthly_df['elev'] # 목표 변수
exog_vars = monthly_df[['temp', 'gtemp', 'pressure']] # 외부 변수

# 데이터 분할 (8:2)
split_point = int(len(monthly_df) * 0.8)
train_endog, test_endog = endog_var[:split_point], endog_var[split_point:]
train_exog, test_exog = exog_vars[:split_point], exog_vars[split_point:]

print(f"훈련 데이터 기간: {train_endog.index.min()} ~ {train_endog.index.max()}")
print(f"테스트 데이터 기간: {test_endog.index.min()} ~ {test_endog.index.max()}")

# 최적 일반 차수 & 최적 계절성 차수
best_order = (1, 1, 0)
best_seasonal_order = (1, 1, 2, 12)


# 모델 학습
print("\nTraining Models...\n")
final_model = sm.tsa.SARIMAX(endog=train_endog, 
                             exog=train_exog, 
                             order=best_order,
                             seasonal_order=best_seasonal_order)
final_result = final_model.fit(disp=False)
print("\nTraining Completed!")

# 테스트 기간에 대해 예측
forecast_values = final_result.forecast(steps=len(test_endog), exog=test_exog)
predicted_values = pd.Series(forecast_values.values, index=test_endog.index)

# 결과 시각화
plt.figure(figsize=(15, 7))

# 실제 값
plt.plot(monthly_df.index, monthly_df['elev'], label='Actual elev')

# 예측 값
plt.plot(predicted_values.index, predicted_values, color='red', linestyle='--', label='Predicted elev')

plt.title(f"Seasonal SARIMAX Model Forecast for code_new = {pilot_code}")
plt.xlabel('Date')
plt.ylabel('elev')
plt.legend()
plt.grid(True)
plt.show()

# 성능 확인
nse_score = cal_nse(test_endog, predicted_values)
kge_score = cal_kge(test_endog, predicted_values)

print("\n성능 평가\n")
print(f"NSE : {nse_score:.4f}")
print(f"KGE : {kge_score:.4f}")