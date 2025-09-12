

# Data_preprogress
# 엑셀 열 이름 바꾸기용!

import pandas as pd

df=pd.read_csv('total_data/2014_2020_시계열_지하수_기상_train.csv',encoding='euc-kr')

df_2=pd.read_csv('total_data/12개_지하수_지점_정보.csv',encoding='euc-kr')



df = df.rename(columns={
    '기온(°C)': 'temp',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'wind',
    '습도(%)': 'humidity',
    '현지기압(hPa)': 'pressure',
    '지면온도(°C)': 'gtemp'
})

df_2 = df_2.rename(columns={
    '수리전도도(cm/sec)': 'welect',
    '표고(EL.m)': 'level'
})



df.to_csv("total_rename_data/2014_2020_시계열_지하수_기상_train.csv", encoding="euc-kr", index=False)

df_2.to_csv("total_rename_data/12개_지하수_지점_정보.csv", encoding="euc-kr", index=False)



