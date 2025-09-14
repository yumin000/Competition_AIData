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

pilot_code = 6
pilot_df = df[df['code_new'] == pilot_code]

# 결측치는 보간으로 채움
if pilot_df.isnull().sum().sum() > 0:
    pilot_df.interpolate(method='time', inplace=True)

endog_var = pilot_df['elev'] # 목표 변수
exog_vars = pilot_df[['temp', 'gtemp', 'pressure']] # 외부 변수

# 데이터 분할 (8:2)
split_point = int(len(pilot_df) * 0.8)
train_endog, test_endog = endog_var[:split_point], endog_var[split_point:]
train_exog, test_exog = exog_vars[:split_point], exog_vars[split_point:]

print(f"훈련 데이터 기간: {train_endog.index.min()} ~ {train_endog.index.max()}")
print(f"테스트 데이터 기간: {test_endog.index.min()} ~ {test_endog.index.max()}")

# 최적 차수 (p, d, q) 찾기 - AIC 기준
p_range = range(1, 4)
d = 0
q_range = range(0, 4)
pdq_combinations = []
for p in p_range:
    for q in q_range:
        pdq_combinations.append((p, d, q))

best_aic = float('inf')
best_order = None

print("\n최적의 SARIMAX 차수를 찾는 중 ...")
for order in pdq_combinations:
    try:
        model = sm.tsa.SARIMAX(endog=train_endog, exog=train_exog, order=order)
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = order
    except Exception as e:
        continue

print(f"최적 차수 (p, d, q) : {best_order} (AIC : {best_aic: .2f})")

# 모델 학습
print("\nTraining Models...\n")
final_model = sm.tsa.SARIMAX(endog=train_endog, exog=train_exog, order=best_order)
final_result = final_model.fit(disp=False)

# 테스트 기간에 대해 예측
forecast_values = final_result.forecast(steps=len(test_endog), exog=test_exog)
predicted_values = pd.Series(forecast_values.values, index=test_endog.index)

# 결과 시각화
plt.figure(figsize=(15, 7))

# 실제 값
plt.plot(pilot_df.index, pilot_df['elev'], label='Actual elev')

# 예측 값
plt.plot(predicted_values.index, predicted_values, color='red', linestyle='--', label='Predicted elev')

plt.title(f"SARIMAX Model Forecast for code_new = {pilot_code}")
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