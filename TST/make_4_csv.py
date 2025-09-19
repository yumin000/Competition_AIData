import pandas as pd



#4번
df=pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv',encoding='euc-kr')
df_4=df[df['code_new']==4]
#df_6=df[df['code_new']==6]
#df_12=df[df['code_new']==12]
df_4.to_csv('code_4.csv',encoding='euc-kr')


