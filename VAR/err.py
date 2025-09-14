import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')
df['ymd'] = pd.to_datetime(df['ymd'])
df.set_index('ymd', inplace=True)

pilot_code = 6
pilot_df = df[df['code_new'] == pilot_code].copy()

if pilot_df.isnull().sum().sum() > 0:
    pilot_df.interpolate(method='time', inplace=True)

endog_var = pilot_df[['elev']] 
exog_vars = pilot_df[['temp', 'gtemp', 'pressure']]

split_point = int(len(pilot_df) * 0.8)
train_endog, test_endog = endog_var[:split_point], endog_var[split_point:]
train_exog, test_exog = exog_vars[:split_point], exog_vars[split_point:]

# --- ✨ 최종 디버깅 파트 ---
print("--- 최종 디버깅 시작 ---")
# 테스트를 위해 몇 개의 order만 시도해봅니다.
orders_to_test = [(1, 0), (1, 1), (2, 1)] 
success_count = 0

for order in orders_to_test:
    print(f"\nTesting Order: {order} ...")
    try:
        model = sm.tsa.SARIMAX(endog=train_endog, exog=train_exog, order=order)
        result = model.fit(disp=False)
        print(f"  -> SUCCESS! (AIC: {result.aic:.2f})")
        success_count += 1
    except Exception as e:
        # 오류 발생 시, 전체 오류 메시지를 출력합니다.
        print(f"  -> FAILED!")
        print("--- ERROR MESSAGE ---")
        raise e # 오류를 직접 발생시켜 전체 Traceback을 확인합니다.
        
if success_count == len(orders_to_test):
    print("\n디버깅 완료: 테스트한 모든 모델이 성공적으로 학습되었습니다.")