import pandas as pd
import statsmodels.api as sm
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

# 데이터 로드 
df_train = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')
df_train['ymd'] = pd.to_datetime(df_train['ymd'])

# 'ymd' 컬럼을 df의 인덱스로 설정
df_train.set_index('ymd', inplace=True)

# 각 관측소의 성능을 저장할 리스트
performance_scores = []

# 1~12번 관측소 함께 학습
for code in range(1, 13):
    print(f"\n>>>{code}번 관측소 훈련 중...")

    station_df = df_train[df_train['code_new']==code].copy()
    monthly_df = station_df.resample('M').mean()

    if code in [9, 10]:
        monthly_df['rainfall_lag1'] = monthly_df['rainfall'].shift(1)
        monthly_df['rainfall_lag2'] = monthly_df['rainfall'].shift(2)
        exog_cols = ['temp', 'gtemp', 'pressure', 'rainfall_lag1', 'rainfall_lag2']

    else:
        exog_cols = ['temp', 'gtemp', 'pressure']

    # .shift()로 인해 생긴 NaN 값을 가진 행 제거 
    monthly_df.dropna(inplace=True)

    # 데이터 분할 (8:2)
    split_point = int(len(monthly_df) * 0.8)
    train_data = monthly_df[:split_point]
    test_data = monthly_df[split_point:]

    endog_train = train_data['elev']
    exog_train = train_data[exog_cols]

    endog_test = test_data['elev']
    exog_test = test_data[exog_cols]

    # 최적 일반 차수 & 최적 계절성 차수
    best_order = (1, 1, 0)
    best_seasonal_order = (1, 1, 2, 12)


    # 모델 학습

    model = sm.tsa.SARIMAX(endog=endog_train, 
                            exog=exog_train, 
                            order=best_order,
                            seasonal_order=best_seasonal_order)
    result = model.fit(disp=False)
    print("\nTraining Completed!")

    # 테스트 기간에 대해 예측
    predictions = result.forecast(steps=len(exog_test), exog=exog_test)
    predicted_values = pd.Series(predictions.values, index=endog_test.index)

    # 성능 확인
    nse_score = cal_nse(endog_test, predicted_values)
    kge_score = cal_kge(endog_test, predicted_values)
    performance_scores.append({'code_new' : code, 'NSE' : nse_score, 'KGE' : kge_score})

    print(f"\n{code}번 관측소 NSE: {nse_score:.4f}, KGE: {kge_score:.4f}")


    # 결과 시각화
    plt.figure(figsize=(10, 5))

    # 실제 값
    plt.plot(monthly_df.index, monthly_df['elev'], label='Actual elev')

    # 예측 값
    plt.plot(predicted_values.index, predicted_values, color='red', linestyle='--', label='Predicted elev')

    plt.title(f"Seasonal SARIMAX Model Forecast for code_new = {code}")
    plt.xlabel('Date')
    plt.ylabel('elev')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\n전체 관측소 성능 요약")
performance_df = pd.DataFrame(performance_scores)
print(performance_df)