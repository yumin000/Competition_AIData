import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('total_rename_data/2014_2020_시계열_지하수_기상_train.csv')

target_df = df[df['code_new'] == 12]

plt.figure(figsize=(15,6))
plt.plot(target_df['ymd'], target_df['elev'], label='Groundwater Lever')
plt.title('Time Series Graph for elec (code_new=12)')
plt.xlabel('Date')
plt.ylabel('Elevation')
plt.grid(True)
plt.legend()
plt.show()