import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # 경고 메시지 무시

import numpy as np
 

# 데이터 로드 
df_train = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')
df_train['ymd'] = pd.to_datetime(df_train['ymd'])

df_test = pd.read_csv('total_rename_data/test_inputs.csv')
df_test['ymd'] = pd.to_datetime(df_test['ymd'])

# 'ymd' 컬럼을 df의 인덱스로 설정
df_train.set_index('ymd', inplace=True)

# 각 관측소의 성능을 저장할 리스트
final_predictions_list = []

# 1~12번 관측소 함께 학습
for code in range(1, 13):
    print(f"\n>>>{code}번 관측소 처리 중...")

    train_station = df_train[df_train['code_new'] == code].copy()
    test_station = df_test[df_test['code_new'] == code].copy()

    train_station.set_index('ymd', inplace=True)
    test_station.set_index('ymd', inplace=True)


    train_monthly = train_station.resample('M').mean()
    test_monthly = test_station.resample('M').mean()

    if code in [9, 10]:
        train_monthly['rainfall_lag1'] = train_monthly['rainfall'].shift(1)
        train_monthly['rainfall_lag2'] = train_monthly['rainfall'].shift(2)
        
        test_monthly['rainfall_lag1'] = test_monthly['rainfall'].shift(1)
        test_monthly['rainfall_lag2'] = test_monthly['rainfall'].shift(2)
        
        exog_cols = ['temp', 'gtemp', 'pressure', 'rainfall_lag1', 'rainfall_lag2']

    else:
        exog_cols = ['temp', 'gtemp', 'pressure']

    # .shift()로 인해 생긴 NaN 값을 가진 행 제거 
    train_monthly.interpolate(method='time', inplace=True)
    train_monthly.dropna(inplace=True)
    test_monthly.interpolate(method='time', inplace=True)
    test_monthly.dropna(inplace=True)

    endog_train = df_train['elev']
    exog_train = df_train[exog_cols]

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
    exog_test = test_monthly[exog_cols]
    predictions = result.forecast(steps=len(exog_test), exog=exog_test)

    # 결과 저장
    pred_df = predictions.to_frame(nema='predicted_elev')
    pred_df['code_new'] = code
    final_predictions_list.append(pred_df)
    print(f"{code}번 관측소 예측 완료!")

# 파일로 저장
final_sub = pd.concat(final_predictions_list)

final_sub = final_sub.pivot_table(
    index=final_sub.index,
    columns='code_new',
    values='predicted_elev'
)

# 컬럼 이름을 문자열로
final_sub.columns = [str(col) for col in final_sub.columns]

# 최종 제출 파일로 저장
final_sub.to_csv('SARIMAX_final_submission.csv')

print(final_sub.head())