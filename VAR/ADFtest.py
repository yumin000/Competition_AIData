import pandas as pd
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')

pilot_code = 6
pilot_df = df[df['code_new'] == pilot_code]

variables_to_test = ['elev', 'temp', 'gtemp', 'pressure']

print(f"---- ADF Test Results for code_new = {pilot_code} ----")

for var in variables_to_test:
    # 결측치 제거
    data_series = pilot_df[var].dropna()

    # ADF Test
    result = adfuller(data_series)

    print(f"\nResults for {var} :")
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}') # p-value

    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}:{value}')