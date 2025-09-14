import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')

pilot_code = 6
pilot_df = df[df['code_new'] == pilot_code]

cols_to_analyze = ['elev', 'wtemp', 'ec', 'temp', 'rainfall',
                   'wind', 'humidity', 'pressure', 'gtemp']
analysis_df = pilot_df[cols_to_analyze]

corr_matrix = analysis_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title(f'Correlation Heatmap for code_new = {pilot_code}')
plt.show()