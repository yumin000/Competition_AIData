import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('total_rename_data/trainData.csv', encoding='cp949')

df['ymd'] = pd.to_datetime(df['ymd'])

df.set_index('ymd', inplace=True)
df.sort_index(inplace=True)

df['elev'].replace(0, np.nan, inplace=True)
df.dropna(inplace=True)
print("결측치 처리 후 elev컬럼의 0 개수 : ", (df['elev']==0).sum())

'''
print("\n데이터 정보 확인")
df.info()

print(df.head())
'''

# 피쳐 엔지니어링 함수
def feature_engineering(df):

    df_copy = df.copy()

    # 시간 피쳐
    df_copy['month'] = df_copy.index.month # 월 1~12
    df_copy['dayofweek'] = df_copy.index.dayofweek # 요일 0:월~6:일
    
    # 지연 피쳐

    # 7일, 14일 전의 지하수위 데이터를 새로운 컬럼으로 추가
    df_copy['groundwater_lag_7'] = df_copy['elev'].shift(7)
    # 1일, 3일 전의 강수량 데이터 추가
    df_copy['rainfall_lag_1'] = df_copy['rainfall'].shift(1)

    # 이동 피쳐

    # (데이터가 부족해도 계산)
    df_copy['rainfall_ma_7'] = df_copy['rainfall'].rolling(window=7, min_periods=1).mean()
    df_copy['rainfall_sum_7'] = df_copy['rainfall'].rolling(window=7, min_periods=1).sum()

    # shift, rolling으로 생긴 맨 앞의 결측치들은 제거해야 모델 학습 가능
    df_copy = df_copy.dropna()

    return df_copy

# 최종 피쳐 엔지니어링 적용
final_df = feature_engineering(df)
print("최종 데이터")
print(final_df.head())


# ------------------------ 데이터 분할 ---------------------------
TARGET = 'elev'

# TARGET 컬럼을 제외하고 모두 입력 특징(X)로 사용
X = final_df.drop(columns=[TARGET])

# TARGET 컬럼만 정답 y 로 사용
y = final_df[TARGET]

# 데이터 분할
split_point = int(len(final_df) * 0.8)

X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

'''
print("X_train의 컬럼 목록")
print(X_train.columns)
'''

# ------------------------ 모델 학습 --------------------------------
# LightGBM 모델
# n_estimators: 모델이 만들 결정 트리의 개수 (클수록 복잡하고 성능은 좋아지지만, 과적합 위험 있음)
# learning_rate: 학습률
# random_state: 결과를 재현하기 위한 값(이 숫자를 고정하면 몇 번을 돌려도 같은 결과가 나옴)

final_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

# 학습 시작
print("\nTraining...")
final_model.fit(X_train, y_train)
print("Comleted!")

# ------------------- 성능 평가 ------------------------
predictions = final_model.predict(X_test)

def get_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

def get_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = np.std(y_pred) / np.std(y_true)
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return kge

nse_score = get_nse(y_test, predictions)
kge_score = get_kge(y_test, predictions)

print("---최종 모델 성능 평가---")
print(f"NSE Score: {nse_score:.4f}")
print(f"KGE Score: {kge_score:.4f}")

# 결과 시각화
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue', width=0.5)))
#fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted', line=dict(color='red', dash='dot', width=0.5), opacity=0.8))
#fig.update_layout(
#    title='실제 값 vs 예측 값',
#    xaxis_title='날짜',
#    yaxis_title='지하수위'
#)

#fig.show()

# 데이터 유출 확인
correlation_matrix = final_df[['elev', 'wtemp', 'ec', 'gtemp']].corr()

print("elev와 주요 피쳐 간 상관 관계\n", correlation_matrix)

import matplotlib.pyplot as plt
from lightgbm import plot_importance

# 학습된 최종 모델의 피처 중요도 시각화
# figsize로 그래프 크기 조절 가능
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(final_model, ax=ax, max_num_features=20) # 상위 20개 피처만 표시
plt.show()